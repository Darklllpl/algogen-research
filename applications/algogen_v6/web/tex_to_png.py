import os
import subprocess
import sys
from pathlib import Path

def compile_tex_to_png(tex_dir, image_dir=None):
    tex_path = Path(tex_dir)
    tex_files = list(tex_path.glob('*.tex'))
    errors = []
    for tex_file in tex_files:
        try:
            # Compile TeX to PDF using xelatex
            subprocess.run(['xelatex', '-output-directory', str(tex_path), str(tex_file)], check=True, cwd=tex_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pdf_file = tex_file.with_suffix('.pdf')
            if pdf_file.exists():
                # Convert PDF to PNG using pdftoppm
                subprocess.run(['pdftoppm', '-png', '-singlefile', str(pdf_file), str(tex_file.stem)], check=True, cwd=tex_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                errors.append(f'PDF not found for {tex_file}')
        except Exception as e:
            errors.append(f'Error processing {tex_file}: {e}')
    if errors:
        for err in errors:
            print(err)
    else:
        print(f'All {len(tex_files)} TeX files converted to PNG successfully.')
        # Move PNG files to target image_dir
        if image_dir:
            target_path = Path(image_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            for tex_file in tex_files:
                png_file = tex_path / f'{tex_file.stem}.png'
                if png_file.exists():
                    new_path = target_path / png_file.name
                    os.replace(str(png_file), str(new_path))

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('Usage: python tex_to_png.py <tex_directory> [image_directory]')
        sys.exit(1)
    image_dir = sys.argv[2] if len(sys.argv) == 3 else None
    compile_tex_to_png(sys.argv[1], image_dir) 