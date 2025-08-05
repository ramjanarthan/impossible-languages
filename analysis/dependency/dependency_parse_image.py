import spacy
import random
from pathlib import Path
from spacy import displacy
import cairosvg
from data_generation.utils.impossible_utils import PERTURBATIONS
from analysis.dependency.dependency_parse import align_tokens_with_tokenizer, apply_windowed_shuffle_perturbation, token_dicts_to_spacy_doc

# --- Configuration ---
nlp = spacy.load("en_core_web_sm")
sentence = "Timothy didn't boast about himself."
OUTPUT_DIR = Path(__file__).parent.parent / "output"
SAVE_AS_PNG = True

# Displacy options for a clean, consistent look (white background, black text)
DISPLACY_OPTIONS = {
    "bg": "#ffffff",
    "color": "#000000",
    "font": "Arial"
}

# --- Functions ---
def save_dependency_parse(doc, file_name):
    """Generates and saves a dependency parse image."""
    svg = displacy.render(doc, style="dep", options=DISPLACY_OPTIONS)
    file_path = OUTPUT_DIR / file_name

    if SAVE_AS_PNG:
        png_path = file_path.with_suffix('.png')
        cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=str(png_path))
        print(f"Saved dependency parse to: {png_path}")
    else:
        svg_path = file_path.with_suffix('.svg')
        svg_path.open("w", encoding="utf-8").write(svg)
        print(f"Saved dependency parse to: {svg_path}")

def save_token_visualization(tokens, file_name):
    """Generates and saves a simple token visualization SVG image."""
    # Create a simple SVG to display tokens in a similar style
    svg_header = '<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="100" style="background-color: {bg}; font-family: {font};">'
    svg_footer = '</svg>'
    text_template = '<text x="{x}" y="50" font-size="20" fill="{color}">{text}</text>'

    x_pos = 20
    svg_content = ""
    for token in tokens:
        svg_content += text_template.format(
            x=x_pos,
            color=DISPLACY_OPTIONS["color"],
            text=token + ","
        )
        x_pos += len(token) * 12 + 15 # Estimate width based on text length

    full_svg = (
        svg_header.format(
            width=x_pos, 
            bg=DISPLACY_OPTIONS["bg"], 
            font=DISPLACY_OPTIONS["font"]
        )
        + svg_content
        + svg_footer
    )

    file_path = OUTPUT_DIR / file_name

    if SAVE_AS_PNG:
        png_path = file_path.with_suffix('.png')
        cairosvg.svg2png(bytestring=full_svg.encode('utf-8'), write_to=str(png_path))
        print(f"Saved token visualization to: {png_path}")
    else:
        svg_path = file_path.with_suffix('.svg')
        svg_path.open("w", encoding="utf-8").write(full_svg)
        print(f"Saved token visualization to: {svg_path}")


def main():
    """Main function to generate and save all three images."""
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # -- Process the original sentence --
    doc = nlp(sentence)

    # -- Save the original dependency parse --
    save_dependency_parse(doc, "dependency_parse_original")

    # -- Visualize the tokenized sentence --
    tokens = [token.text for token in doc]
    save_token_visualization(tokens, "token_visualization")

    # -- Visualize the GPT-2 tokenized sentence --
    tokenizer = PERTURBATIONS["shuffle_control"]["gpt2_tokenizer"]
    gpt_2_tokenized = tokenizer.encode(sentence)
    gpt_2_tokens = [tokenizer.decode(token) for token in gpt_2_tokenized]
    save_token_visualization(gpt_2_tokens, "gpt_2_token_visualization")

    # -- Align the GPT-2 tokens with the original sentence --
    aligned_tokens = align_tokens_with_tokenizer(sentence, doc, tokenizer)
    shuffled_token_dicts = apply_windowed_shuffle_perturbation(aligned_tokens, window=3, seed=0)

    # -- Visualize the shuffled tokens --
    shuffled_tokens = [token['text'] for token in shuffled_token_dicts]
    save_token_visualization(shuffled_tokens, "gpt_2_token_shuffled_visualization")

    # -- Visualize the shuffled dependency parse --
    shuffled_doc = token_dicts_to_spacy_doc(shuffled_token_dicts, nlp)
    save_dependency_parse(shuffled_doc, "dependency_parse_shuffled")

if __name__ == "__main__":
    main()