use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(name = "llama-cli")]
#[command(about = "Command-line interface for llama-rs")]
struct Args {
    /// Model file path
    #[arg(short, long)]
    model: String,
}

fn main() -> Result<()> {
    let _args = Args::parse();
    
    println!("llama-cli: Not yet implemented");
    println!("This is a placeholder for the future CLI tool");
    
    Ok(())
}
