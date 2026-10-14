
import argparse
import torch

from colorama import init
from transformers import AutoTokenizer, AutoModelForCausalLM

from configuration import GPT2Config
from gpt2 import GPT2LMHeadModel


init(autoreset=True)

MODEL_NAME = "openai-community/gpt2-medium"

# Your original HEX color palette
HEX_INFO = "8BE9FD"
HEX_USER = "FF79C6"
HEX_RESP = "F1FA8C"
HEX_MUTED = "6272A4"


def hex_to_ansi(hex_str):
    hex_str = hex_str.lstrip("#")
    r, g, b = tuple(
        int(hex_str[i:i + 2], 16)
        for i in (0, 2, 4)
    )
    return f"\x1b[38;2;{r};{g};{b}m"


ANSI_INFO = hex_to_ansi(HEX_INFO)
ANSI_USER = hex_to_ansi(HEX_USER)
ANSI_RESP = hex_to_ansi(HEX_RESP)
ANSI_MUTED = hex_to_ansi(HEX_MUTED)
ANSI_RESET = "\x1b[0m"


# Your original printing functions
def print_info(text):
    print(f"{ANSI_MUTED}• {ANSI_INFO}{text}{ANSI_RESET}")


def print_response(text):
    print(f"{ANSI_MUTED}  ↳ {ANSI_RESP}{text}{ANSI_RESET}\n")


def get_user_input(prompt_label):
    try:
        prompt = (
            f"{ANSI_MUTED}["
            f"{ANSI_INFO}{prompt_label}"
            f"{ANSI_MUTED}] ❯ {ANSI_USER}"
        )

        user_text = input(prompt)
        print(ANSI_RESET, end="")

        return user_text

    except (KeyboardInterrupt, EOFError):
        print(ANSI_RESET)
        return "exit"


def load_model(test_mode=False):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print_info(f"Device: {device}")
    print_info("Loading Tokenizer & Model")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print_info("Tokenizer: Loaded!")

    if test_mode:
        print_info("TEST MODE: Tiny GPT-2 with random weights")

        config = GPT2Config(
            block_size=128,
            n_layers=2,
            n_heads=4,
            n_embd=128,
            eos_token_id=tokenizer.eos_token_id,
        )

        model = GPT2LMHeadModel(config)

    else:
        print_info("Loading Hugging Face GPT-2 Medium")

        # Load pretrained Hugging Face model on CPU first
        hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            attn_implementation="eager",
        )

        config = GPT2Config(
            eos_token_id=tokenizer.eos_token_id
        )

        # Your custom implementation
        model = GPT2LMHeadModel(config)

        # Copy pretrained weights into your model
        model.load_state_dict(
            hf_model.state_dict(),
            strict=True
        )

        print_info("Pretrained weights copied successfully!")

        # Release the Hugging Face model
        del hf_model

    model = model.to(device)
    model.eval()

    print_info("Model: Loaded!")

    return model, tokenizer, device


def test_forward_pass(model, device):
    print_info("Running forward-pass test...")

    idx = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(2, 8),
        device=device,
    )

    with torch.no_grad():
        logits = model(idx)

    expected_shape = (2, 8, model.config.vocab_size)

    assert logits.shape == expected_shape, (
        f"Expected {expected_shape}, got {logits.shape}"
    )

    print_info(f"Forward pass: PASSED {tuple(logits.shape)}")


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a tiny GPT-2 with random weights",
    )

    args = parser.parse_args()

    model, tokenizer, device = load_model(
        test_mode=args.test
    )

    test_forward_pass(model, device)

    if args.test:
        print_info(
            "Random weights: generated text will be meaningless."
        )

    while True:
        user_input = get_user_input("USER")
        commands = user_input.strip().lower()

        if commands in ["/exit", "exit", "/quit", "quit"]:
            print_info("Exiting !!!")
            break

        if not user_input.strip():
            continue

        encoded_input = tokenizer.encode(
            user_input,
            return_tensors="pt"
        ).to(device)

        encoded_response = model.generate_text(
            encoded_input,
            max_tokens=30 if args.test else 100
        )

        # Preserve your original behavior:
        # display the prompt AND generated continuation
        response = tokenizer.decode(
            encoded_response[0],
            skip_special_tokens=True
        )

        print_response(response)


if __name__ == "__main__":
    run()
