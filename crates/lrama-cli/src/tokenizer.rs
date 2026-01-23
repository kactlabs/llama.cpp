//! Tokenizer for LLaMA models
//!
//! Implements BPE (Byte Pair Encoding) tokenization compatible with GGUF models.

#![allow(dead_code)]

use anyhow::{Context, Result};
use gguf::{GGUFReader, MetadataValue};
use std::collections::HashMap;

#[derive(Debug)]
pub struct Tokenizer {
    /// Token string to ID mapping
    vocab: HashMap<String, u32>,
    /// ID to token string mapping
    id_to_token: Vec<String>,
    /// Special tokens
    pub bos_token: u32,
    pub eos_token: u32,
    pub nl_token: Option<u32>,
}

impl Tokenizer {
    /// Load tokenizer from GGUF metadata
    pub fn from_gguf(reader: &GGUFReader) -> Result<Self> {
        // Load tokens array
        let tokens = reader.get_metadata("tokenizer.ggml.tokens")
            .context("Missing tokenizer.ggml.tokens")?;
        
        let token_strings = match tokens {
            MetadataValue::Array(arr) => {
                arr.values.iter()
                    .map(|v| match v {
                        MetadataValue::String(s) => Ok(s.clone()),
                        _ => Err(anyhow::anyhow!("Expected string token")),
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            _ => return Err(anyhow::anyhow!("tokens must be an array")),
        };
        
        // Build vocab mappings
        let mut vocab = HashMap::new();
        for (id, token) in token_strings.iter().enumerate() {
            vocab.insert(token.clone(), id as u32);
        }
        
        // Load special tokens
        let bos_token = reader.get_metadata("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32().ok())
            .unwrap_or(1);
        
        let eos_token = reader.get_metadata("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32().ok())
            .unwrap_or(2);
        
        // Try to find newline token
        let nl_token = vocab.get("\n").or_else(|| vocab.get("\\n")).copied();
        
        Ok(Self {
            vocab,
            id_to_token: token_strings,
            bos_token,
            eos_token,
            nl_token,
        })
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        if add_bos {
            tokens.push(self.bos_token);
        }
        
        // Simple greedy tokenization (not true BPE, but works for demo)
        // For production, would need proper BPE merge rules
        let mut remaining = text;
        
        while !remaining.is_empty() {
            // Try to match longest token
            let mut matched = false;
            for len in (1..=remaining.len()).rev() {
                let substr = &remaining[..len];
                if let Some(&token_id) = self.vocab.get(substr) {
                    tokens.push(token_id);
                    remaining = &remaining[len..];
                    matched = true;
                    break;
                }
            }
            
            // If no match, try byte fallback
            if !matched {
                let byte = remaining.as_bytes()[0];
                let byte_token = format!("<0x{:02X}>", byte);
                if let Some(&token_id) = self.vocab.get(&byte_token) {
                    tokens.push(token_id);
                } else {
                    // Unknown token - use a fallback
                    tokens.push(0); // Usually <unk>
                }
                remaining = &remaining[1..];
            }
        }
        
        tokens
    }
    
    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();
        
        for &token_id in tokens {
            if token_id == self.bos_token || token_id == self.eos_token {
                continue; // Skip special tokens
            }
            
            if let Some(token_str) = self.id_to_token.get(token_id as usize) {
                // Handle byte tokens
                if token_str.starts_with("<0x") && token_str.ends_with(">") {
                    if let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16) {
                        result.push(byte as char);
                    }
                } else {
                    result.push_str(token_str);
                }
            }
        }
        
        result
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenizer_basic() {
        // Would need a real GGUF file to test properly
        // This is a placeholder for the structure
    }
}
