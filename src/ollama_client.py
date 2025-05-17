import subprocess

def generate_with_ollama(prompt: str,
                         model: str = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
                         timeout: int = 60) -> str:
    """
    Calls `ollama run <model>` and feeds it the prompt via stdin.
    Forces UTF-8 decoding (replacing errors) so we never get a decode crash.
    """
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",      # replace any invalid bytes
        timeout=timeout
    )
    if proc.returncode != 0:
        # stderr might also be None if it failed catastrophically
        err = proc.stderr or "<no stderr>"
        raise RuntimeError(f"Ollama error: {err.strip()}")
    # stdout should be a string (possibly with � replacements)
    return (proc.stdout or "").strip()
