// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use indexmap::IndexMap;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};

use crate::protocol::handshake::{EngineCoreReadyResponse, FrontendHashConfig};
use crate::protocol::request::EngineCoreRequest;

const BLOCK_SIZE: usize = 32;
const MEMO_CHECKPOINT: usize = 1024;
const MAX_ENTRIES: usize = 512;
const MAX_CACHED_TOKENS: usize = 16 * 1024 * 1024;
const MAX_SALT_BYTES: usize = 4096;
type Hash = [u8; 32];

#[derive(Clone, Eq, Hash, PartialEq)]
struct MemoKey {
    salt: Option<Arc<str>>,
    tokens: usize,
    digest: Hash,
}

struct MemoEntry {
    tokens: Vec<u32>,
    hashes: Vec<Hash>,
    keys: Vec<MemoKey>,
}

#[derive(Default)]
struct Memo {
    entries: IndexMap<u64, Arc<MemoEntry>>,
    prefixes: HashMap<MemoKey, u64>,
    tokens: usize,
    next_id: u64,
}

impl Memo {
    fn find(&mut self, keys: &[MemoKey]) -> Option<(usize, Arc<MemoEntry>)> {
        for key in keys.iter().rev() {
            if let Some(&id) = self.prefixes.get(key) {
                let entry = self.entries.shift_remove(&id)?;
                self.entries.insert(id, Arc::clone(&entry));
                return Some((key.tokens, entry));
            }
        }
        None
    }

    fn insert(&mut self, entry: Arc<MemoEntry>) -> Vec<Arc<MemoEntry>> {
        let id = self.next_id;
        self.next_id += 1;
        for key in &entry.keys {
            self.prefixes.insert(key.clone(), id);
        }
        self.tokens += entry.tokens.len();
        self.entries.insert(id, entry);
        let mut evicted = Vec::new();
        while self.entries.len() > MAX_ENTRIES || self.tokens > MAX_CACHED_TOKENS {
            if let Some((old_id, old)) = self.entries.shift_remove_index(0) {
                self.tokens -= old.tokens.len();
                for key in &old.keys {
                    if self.prefixes.get(key) == Some(&old_id) {
                        self.prefixes.remove(key);
                    }
                }
                evicted.push(old);
            }
        }
        evicted
    }
}

fn memo_keys(tokens: &[u32], salt: Option<&str>) -> Vec<MemoKey> {
    let salt = salt.map(Arc::from);
    let mut digest = Sha256::new();
    let mut keys = Vec::with_capacity(tokens.len() / MEMO_CHECKPOINT);
    let mut bytes = [0; MEMO_CHECKPOINT * 4];
    for (index, checkpoint) in tokens.chunks_exact(MEMO_CHECKPOINT).enumerate() {
        for (token, bytes) in checkpoint.iter().zip(bytes.chunks_exact_mut(4)) {
            bytes.copy_from_slice(&token.to_le_bytes());
        }
        digest.update(bytes);
        keys.push(MemoKey {
            salt: salt.clone(),
            tokens: (index + 1) * MEMO_CHECKPOINT,
            digest: digest.clone().finalize().into(),
        });
    }
    keys
}

pub(crate) struct FrontendHasher {
    none_hash: Hash,
    memo: Mutex<Memo>,
    hashed: AtomicU64,
    skipped: AtomicU64,
    blocks: AtomicU64,
    reused: AtomicU64,
}

impl FrontendHasher {
    pub(crate) fn from_ready_responses<'a>(
        ready: impl Iterator<Item = &'a EngineCoreReadyResponse>,
    ) -> Option<Arc<Self>> {
        let enabled = std::env::var("VLLM_NATIVE_FEH");
        if enabled.as_deref() != Ok("1") {
            if enabled.as_deref() == Ok("0") {
                tracing::info!("native FEH disabled: VLLM_NATIVE_FEH=0");
            }
            return None;
        }
        let configs: Option<Vec<_>> =
            ready.map(|ready| ready.native_frontend_hash_config.as_ref()).collect();
        let hasher = configs.and_then(|configs| {
            let config = *configs.first()?;
            configs.iter().all(|other| *other == config).then_some(())?;
            Self::new(config)
        });
        if hasher.is_some() {
            tracing::info!(
                block_size = BLOCK_SIZE,
                "native FEH enabled: sha256-pickle5-feh1"
            );
        } else {
            tracing::warn!("native FEH disabled: missing or unsupported engine hash metadata");
        }
        hasher.map(Arc::new)
    }

    fn new(config: &FrontendHashConfig) -> Option<Self> {
        if config.algorithm != "sha256-pickle5-feh1" || config.block_size != BLOCK_SIZE as u32 {
            return None;
        }
        let none_hash = hex::decode(&config.none_hash).ok()?.try_into().ok()?;
        Some(Self {
            none_hash,
            memo: Mutex::default(),
            hashed: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            blocks: AtomicU64::new(0),
            reused: AtomicU64::new(0),
        })
    }

    pub(crate) fn attach(&self, request: &mut EngineCoreRequest) {
        request.prompt_block_hashes = None;
        let Some(tokens) = request.prompt_token_ids.as_deref() else {
            self.skipped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if request.mm_features.as_ref().is_some_and(|features| !features.is_empty())
            || request.lora_request.is_some()
            || request.prompt_embeds.is_some()
            || request.prompt_is_token_ids.is_some()
            || request.cache_salt.as_ref().is_some_and(|salt| salt.len() > MAX_SALT_BYTES)
        {
            self.skipped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let tokens = &tokens[..tokens.len() / BLOCK_SIZE * BLOCK_SIZE];
        let keys = memo_keys(tokens, request.cache_salt.as_deref());
        let cached = self.memo.lock().find(&keys);
        let cached = cached
            .filter(|(length, entry)| entry.tokens.get(..*length) == Some(&tokens[..*length]));
        let reused = cached.as_ref().map_or(0, |(length, _)| length / BLOCK_SIZE);
        let mut hashes = Vec::with_capacity(tokens.len() / BLOCK_SIZE);
        if let Some((_, entry)) = &cached {
            hashes.extend_from_slice(&entry.hashes[..reused]);
        }
        let mut parent = hashes.last().copied().unwrap_or(self.none_hash);
        let mut pickle = Vec::with_capacity(256);
        for (index, block) in tokens.chunks_exact(BLOCK_SIZE).enumerate().skip(reused) {
            let salt = (index == 0).then_some(request.cache_salt.as_deref()).flatten();
            parent = hash_block(&mut pickle, &parent, block, salt);
            hashes.push(parent);
        }
        let mut blob = Vec::with_capacity(8 + hashes.len() * 32);
        blob.extend_from_slice(b"FEH1");
        blob.extend_from_slice(&(BLOCK_SIZE as u32).to_le_bytes());
        for hash in &hashes {
            blob.extend_from_slice(hash);
        }
        let blocks = hashes.len() as u64;
        if !keys.is_empty() && tokens.len() <= MAX_CACHED_TOKENS {
            let entry = Arc::new(MemoEntry {
                tokens: tokens.to_vec(),
                hashes,
                keys,
            });
            let evicted = self.memo.lock().insert(entry);
            drop(evicted);
        }
        request.prompt_block_hashes = Some(blob);
        self.blocks.fetch_add(blocks, Ordering::Relaxed);
        self.reused.fetch_add(reused as u64, Ordering::Relaxed);
        let hashed = self.hashed.fetch_add(1, Ordering::Relaxed) + 1;
        if hashed % 2000 == 1 {
            tracing::info!(
                hashed,
                skipped = self.skipped.load(Ordering::Relaxed),
                blocks = self.blocks.load(Ordering::Relaxed),
                reused = self.reused.load(Ordering::Relaxed),
                "native FEH stats"
            );
        }
    }
}

// CPython pickle protocol 5 for (bytes[32], tuple[u32; 32], optional tuple[str]).
fn hash_block(buffer: &mut Vec<u8>, parent: &Hash, tokens: &[u32], salt: Option<&str>) -> Hash {
    buffer.clear();
    buffer.extend_from_slice(b"\x80\x05\x95\0\0\0\0\0\0\0\0C ");
    buffer.extend_from_slice(parent);
    buffer.extend_from_slice(b"\x94(");
    for &token in tokens {
        if token <= u8::MAX as u32 {
            buffer.extend_from_slice(&[b'K', token as u8]);
        } else if token <= u16::MAX as u32 {
            buffer.push(b'M');
            buffer.extend_from_slice(&(token as u16).to_le_bytes());
        } else if token <= i32::MAX as u32 {
            buffer.push(b'J');
            buffer.extend_from_slice(&token.to_le_bytes());
        } else {
            buffer.extend_from_slice(b"\x8a\x05");
            buffer.extend_from_slice(&token.to_le_bytes());
            buffer.push(0);
        }
    }
    buffer.extend_from_slice(b"t\x94");
    if let Some(salt) = salt.filter(|salt| !salt.is_empty()) {
        if salt.len() <= u8::MAX as usize {
            buffer.extend_from_slice(&[0x8c, salt.len() as u8]);
        } else {
            buffer.push(b'X');
            buffer.extend_from_slice(&(salt.len() as u32).to_le_bytes());
        }
        buffer.extend_from_slice(salt.as_bytes());
        buffer.extend_from_slice(b"\x94\x85\x94");
    } else {
        buffer.push(b'N');
    }
    buffer.extend_from_slice(b"\x87\x94.");
    let frame_len = (buffer.len() - 11) as u64;
    buffer[3..11].copy_from_slice(&frame_len.to_le_bytes());
    Sha256::digest(buffer).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    const NONE_HASH: &str = "066b1e8987e027138e98701c46128929378935f9b6cd1549105ae6b4f0d71198";

    fn hasher(seed: &str) -> FrontendHasher {
        FrontendHasher::new(&FrontendHashConfig {
            algorithm: "sha256-pickle5-feh1".into(),
            block_size: 32,
            none_hash: seed.into(),
        })
        .unwrap()
    }

    fn tokens() -> Vec<u32> {
        [0, 255, 256, 65535, 65536, 2147483647, 2147483648, u32::MAX].repeat(8)
    }

    #[test]
    fn prompt_hashes_match_cpython_pickle5_with_partial_blocks() {
        let hasher = hasher(NONE_HASH);
        let mut request = EngineCoreRequest {
            prompt_token_ids: Some(tokens()[..33].to_vec()),
            ..EngineCoreRequest::default()
        };
        hasher.attach(&mut request);
        assert_eq!(
            hex::encode(request.prompt_block_hashes.as_ref().unwrap()),
            concat!(
                "4645483120000000",
                "645c0806aa088e851eda7e4226eb128f8d40a64fe5d0949d243686cd9b5a3431",
            )
        );
        request.prompt_token_ids = Some(tokens());
        hasher.attach(&mut request);
        assert_eq!(
            hex::encode(request.prompt_block_hashes.as_ref().unwrap()),
            concat!(
                "4645483120000000",
                "645c0806aa088e851eda7e4226eb128f8d40a64fe5d0949d243686cd9b5a3431",
                "a9cae95b149ac68f3f40b9badb691c2a549f35376fc48beb4a2296506bca5eb3",
            )
        );
        assert_eq!(hasher.reused.load(Ordering::Relaxed), 0);
        let expected = request.prompt_block_hashes.clone();
        request.prompt_token_ids.as_mut().unwrap()[31] = 43;
        hasher.attach(&mut request);
        assert_eq!(
            hex::encode(&request.prompt_block_hashes.as_ref().unwrap()[8..]),
            concat!(
                "1e4d90fcf11fdaf58f44d3fe09dd741ff22606b8686ea74dd4621fc99c252fda",
                "a12d664ade56926022914b24666b0d9eecf50765b8a0a94608815a98970854c6",
            )
        );
        request.prompt_token_ids = Some(tokens());
        request.cache_salt = Some(String::new());
        hasher.attach(&mut request);
        assert_eq!(request.prompt_block_hashes, expected);
    }

    #[test]
    fn memo_reuses_interleaved_conversations_and_only_matching_salted_prefixes() {
        let memoized = hasher(NONE_HASH);
        let mut request = EngineCoreRequest {
            prompt_token_ids: Some(vec![11; MEMO_CHECKPOINT * 3]),
            ..EngineCoreRequest::default()
        };
        memoized.attach(&mut request);
        let mut other = EngineCoreRequest {
            prompt_token_ids: Some(vec![22; MEMO_CHECKPOINT * 3]),
            ..EngineCoreRequest::default()
        };
        memoized.attach(&mut other);

        request.prompt_token_ids.as_mut().unwrap().extend([33; MEMO_CHECKPOINT + 7]);
        memoized.attach(&mut request);
        let mut cold = request.clone();
        hasher(NONE_HASH).attach(&mut cold);
        assert_eq!(request.prompt_block_hashes, cold.prompt_block_hashes);
        let reused = (MEMO_CHECKPOINT * 3 / BLOCK_SIZE) as u64;
        assert_eq!(memoized.reused.load(Ordering::Relaxed), reused);

        request.prompt_token_ids.as_mut().unwrap()[MEMO_CHECKPOINT * 2] = 44;
        memoized.attach(&mut request);
        cold = request.clone();
        hasher(NONE_HASH).attach(&mut cold);
        assert_eq!(request.prompt_block_hashes, cold.prompt_block_hashes);
        let reused = reused + (MEMO_CHECKPOINT * 2 / BLOCK_SIZE) as u64;
        assert_eq!(memoized.reused.load(Ordering::Relaxed), reused);

        let unsalted = request.prompt_block_hashes.clone();
        request.cache_salt = Some("tenant".into());
        memoized.attach(&mut request);
        cold = request.clone();
        hasher(NONE_HASH).attach(&mut cold);
        assert_eq!(request.prompt_block_hashes, cold.prompt_block_hashes);
        assert_ne!(request.prompt_block_hashes, unsalted);
        assert_eq!(memoized.reused.load(Ordering::Relaxed), reused);
    }

    #[test]
    fn memo_eviction_keeps_newer_shared_checkpoints_and_bounds_the_index() {
        let memoized = hasher(NONE_HASH);
        let mut request = EngineCoreRequest::default();
        for i in 0..=MAX_ENTRIES {
            let mut tokens = vec![42; MEMO_CHECKPOINT];
            tokens.extend(vec![i as u32; MEMO_CHECKPOINT]);
            request.prompt_token_ids = Some(tokens);
            memoized.attach(&mut request);
        }
        {
            let memo = memoized.memo.lock();
            assert_eq!(memo.entries.len(), MAX_ENTRIES);
            assert_eq!(memo.prefixes.len(), MAX_ENTRIES + 1);
            assert_eq!(memo.tokens, MAX_ENTRIES * MEMO_CHECKPOINT * 2);
        }
        let reused = memoized.reused.load(Ordering::Relaxed);
        request.prompt_token_ids.as_mut().unwrap()[MEMO_CHECKPOINT..].fill(0);
        memoized.attach(&mut request);
        let mut cold = request.clone();
        hasher(NONE_HASH).attach(&mut cold);
        assert_eq!(request.prompt_block_hashes, cold.prompt_block_hashes);
        assert_eq!(
            memoized.reused.load(Ordering::Relaxed) - reused,
            (MEMO_CHECKPOINT / BLOCK_SIZE) as u64,
        );
    }

    #[test]
    fn memo_checks_exact_tokens_before_reusing_a_fingerprint_match() {
        let memoized = hasher(NONE_HASH);
        let mut request = EngineCoreRequest {
            prompt_token_ids: Some(vec![42; MEMO_CHECKPOINT]),
            ..EngineCoreRequest::default()
        };
        memoized.attach(&mut request);
        let expected = request.prompt_block_hashes.clone();
        {
            let mut memo = memoized.memo.lock();
            let entry = memo.entries.get_mut(&0).unwrap();
            Arc::get_mut(entry).unwrap().tokens[0] = 43;
        }
        memoized.attach(&mut request);
        assert_eq!(request.prompt_block_hashes, expected);
        assert_eq!(memoized.reused.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn salt_and_engine_seed_match_cpython_pickle5() {
        for (seed, salt, expected) in [
            (
                NONE_HASH,
                "é".repeat(150),
                concat!(
                    "c348e02638df5cad09ffc2b12df5ae3545c9311c4d9122b45b6809d5c71ab195",
                    "9d14a1b43cbf4bb9569724b27f7abd13a8f4481c6b3819303401dcf8fa4e6a4d",
                ),
            ),
            (
                "8508caabf15277770fec9d07df755897d40add4758a1ca33355c8dc7e8525c69",
                "é".repeat(150),
                concat!(
                    "ff5d867846530e965b2f7aa3be6515a4d7e80c47c68f254f54ae0c17ea4b59b2",
                    "f6205da99562f64b19f98fd97e900956d964b67b52d2a2ffbc05ebc61629afdc",
                ),
            ),
            (
                NONE_HASH,
                "salt".into(),
                concat!(
                    "5c1124490c77cccf822464addc47d73090bf7646dd6f3f9ffe1da4f570111be1",
                    "c2e20609fec4b717600702ce4f52ea899e75b1213cd4cdac43b597de01fc570a",
                ),
            ),
        ] {
            let mut request = EngineCoreRequest {
                prompt_token_ids: Some(tokens()),
                cache_salt: Some(salt),
                ..EngineCoreRequest::default()
            };
            hasher(seed).attach(&mut request);
            assert_eq!(
                hex::encode(&request.prompt_block_hashes.unwrap()[8..]),
                expected
            );
        }
    }

    #[test]
    fn unsupported_inputs_and_config_fall_back_to_engine_hashing() {
        let hasher = hasher(NONE_HASH);
        let mut request = EngineCoreRequest {
            prompt_token_ids: Some(tokens()),
            prompt_is_token_ids: Some(vec![true; 64]),
            ..EngineCoreRequest::default()
        };
        hasher.attach(&mut request);
        assert!(request.prompt_block_hashes.is_none());
        request.prompt_is_token_ids = None;
        request.cache_salt = Some("a".repeat(MAX_SALT_BYTES + 1));
        hasher.attach(&mut request);
        assert!(request.prompt_block_hashes.is_none());
        assert!(
            FrontendHasher::new(&FrontendHashConfig {
                algorithm: "sha256_cbor".into(),
                block_size: 32,
                none_hash: NONE_HASH.into(),
            })
            .is_none()
        );
    }
}
