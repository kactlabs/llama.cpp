//! lrama-cli: Rust-based LLaMA inference CLI
//!
//! A pure Rust implementation of LLaMA model inference using GGUF format.

mod tokenizer;
mod model;
mod kv_cache;
mod attention;
mod inference;
mod generate;

use anyhow::{Context, Result};
use clap::Parser;
use gguf::GGUFReader;
use std::path::PathBuf;
use tokenizer::Tokenizer;
use model::LlamaModel;
use ggml_core::Context as GGMLContext;
use generate::{generate, GenerationConfig};

#[derive(Parser, Debug)]
#[command(name = "lrama-cli")]
#[command(about = "Rust-based LLaMA inference CLI", long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: PathBuf,

    /// Prompt text
    #[arg(short, long)]
    prompt: Option<String>,

    /// Number of tokens to generate
    #[arg(short = 'n', long, default_value = "128")]
    n_predict: usize,

    /// Temperature for sampling
    #[arg(short, long, default_value = "0.8")]
    temperature: f32,

    /// Show model information only
    #[arg(long)]
    info: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("🦙 lrama-cli - Rust LLaMA Inference");
    println!("=====================================\n");

    // Load GGUF model
    println!("Loading model: {}", args.model.display());
    let reader = GGUFReader::open(&args.model)
        .context("Failed to open GGUF file")?;

    // Display model information
    print_model_info(&reader)?;

    if args.info {
        return Ok(());
    }

    // Load tokenizer
    println!("\n📚 Loading tokenizer...");
    let tokenizer = Tokenizer::from_gguf(&reader)?;
    println!("  Vocabulary size: {}", tokenizer.vocab_size());
    println!("  BOS token: {}", tokenizer.bos_token);
    println!("  EOS token: {}", tokenizer.eos_token);

    // Load model
    println!("\n🧠 Loading model weights...");
    let mem_size = 8 * 1024 * 1024 * 1024; // 8 GB
    let mut ctx = GGMLContext::new(mem_size)?;
    let model = LlamaModel::from_gguf(&reader, &mut ctx)?;

    // Get prompt
    let prompt = args.prompt.unwrap_or_else(|| {
        "Once upon a time".to_string()
    });

    println!("\n📝 Prompt: {}", prompt);
    println!("🎲 Temperature: {}", args.temperature);
    println!("🔢 Generating {} tokens\n", args.n_predict);

    // Tokenize
    println!("🔤 Tokenizing...");
    let tokens = tokenizer.encode(&prompt, true);
    println!("  {} tokens: {:?}", tokens.len(), &tokens[..tokens.len().min(10)]);

    // Generate text!
    println!("\n🎯 Generating text...");
    let config = GenerationConfig {
        temperature: args.temperature,
        n_predict: args.n_predict,
        ..Default::default()
    };
    
    let generated = generate(&model, &tokenizer, &prompt, &config)?;
    
    println!("\n📄 Generated text:");
    println!("─────────────────────────────────────");
    println!("{}", generated);
    println!("─────────────────────────────────────");

    Ok(())
}

fn print_model_info(reader: &GGUFReader) -> Result<()> {
    println!("📊 Model Information:");
    println!("   Version: {:?}", reader.version());
    println!("   Tensors: {}", reader.tensor_count());

    // Print key metadata
    if let Some(name) = reader.get_metadata("general.name") {
        println!("   Name: {:?}", name);
    }
    if let Some(arch) = reader.get_metadata("general.architecture") {
        println!("   Architecture: {:?}", arch);
    }
    
    // Print all metadata keys for debugging
    println!("\n🔍 Metadata keys:");
    let metadata = reader.metadata();
    let mut keys: Vec<_> = metadata.keys().collect();
    keys.sort();
    for key in keys.iter().take(20) {
        println!("   {}", key);
    }
    if keys.len() > 20 {
        println!("   ... and {} more", keys.len() - 20);
    }

    // Print some tensor names
    let tensor_names = reader.tensor_names();
    println!("\n📦 Sample tensors ({} total):", tensor_names.len());
    for (i, name) in tensor_names.iter().take(10).enumerate() {
        if let Some(info) = reader.get_tensor_info(name) {
            println!("   {}. {} - {:?} {:?}", 
                     i + 1, name, info.tensor_type, info.dimensions);
        }
    }
    if tensor_names.len() > 10 {
        println!("   ... and {} more", tensor_names.len() - 10);
    }

    Ok(())
}
