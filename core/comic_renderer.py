import os
from PIL import Image, ImageDraw, ImageFont

class ComicRenderer:
    def __init__(self):
        # We try to use a standard font, fallback to default if not found
        self.font_path = self._get_default_font()
        
    def _get_default_font(self):
        # Try to find a standard Windows font (Arial)
        windows_font = "C:\\Windows\\Fonts\\arialbd.ttf" # Bold font looks better for comics
        if not os.path.exists(windows_font):
            windows_font = "C:\\Windows\\Fonts\\arial.ttf"
        if os.path.exists(windows_font):
            return windows_font
        return None

    def _wrap_text(self, text: str, font, max_width: int, draw: ImageDraw) -> list:
        lines = []
        for manual_line in text.split('\n'):
            words = manual_line.split()
            current_line = []
            
            for word in words:
                current_line.append(word)
                bbox = draw.textbbox((0, 0), " ".join(current_line), font=font)
                if bbox[2] - bbox[0] > max_width:
                    current_line.pop()
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(" ".join(current_line))
        return lines

    def draw_speech_bubble(self, image_path: str, dialogues: list, output_path: str):
        """Draws professional manga-style speech bubbles and narration boxes."""
        if not dialogues:
            return image_path
            
        try:
            img = Image.open(image_path).convert("RGBA")
            overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            if self.font_path:
                try:
                    font = ImageFont.truetype(self.font_path, 16)
                except IOError:
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()
                
            y_offset = 20
            # Alternate positions to simulate comic flow
            positions = ['top-left', 'bottom-right', 'top-right', 'bottom-left']
            
            for i, dialogue in enumerate(dialogues):
                pos = positions[i % len(positions)]
                
                speaker = dialogue.get("speaker", "Unknown")
                dlg_type = dialogue.get("type", "speech").lower()
                
                # Format text
                raw_text = dialogue.get('text', '').strip()
                # Clean up any accidental leading colons or quotes from LLM
                while raw_text.startswith(':') or raw_text.startswith('"'):
                    raw_text = raw_text[1:].strip()
                while raw_text.endswith('"'):
                    raw_text = raw_text[:-1].strip()

                if dlg_type == "narration" or speaker.lower() == "narrator":
                    text = raw_text
                    is_narration = True
                else:
                    text = f"{speaker}\n{raw_text}"
                    is_narration = False
                
                # Wrap Text
                max_width = 220 if is_narration else 180
                lines = self._wrap_text(text, font, max_width, draw)
                
                if not lines:
                    continue
                    
                line_height = draw.textbbox((0, 0), lines[0], font=font)[3] - draw.textbbox((0, 0), lines[0], font=font)[1]
                text_width = max([draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0] for line in lines])
                
                padding_x = 25
                padding_y = 20
                bubble_width = text_width + (padding_x * 2)
                bubble_height = (line_height * len(lines)) + (padding_y * 2) + (len(lines) * 4)
                
                # Calculate Coordinates based on position
                if pos.startswith('top'):
                    y = y_offset
                    y_offset += bubble_height + 40
                else:
                    y = img.height - bubble_height - 30
                    
                if pos.endswith('left'):
                    x = 20
                else:
                    x = img.width - bubble_width - 20
                
                # Ensure it doesn't go out of bounds
                x = max(10, min(x, img.width - bubble_width - 10))
                y = max(10, min(y, img.height - bubble_height - 10))
                
                if is_narration:
                    # NARRATION BOX (Rectangular, black/dark background, white text)
                    box_fill = (20, 20, 20, 230)
                    text_fill = (255, 255, 255, 255)
                    outline_fill = (255, 255, 255, 200)
                    
                    # Draw Box
                    draw.rectangle([x, y, x + bubble_width, y + bubble_height], fill=box_fill, outline=outline_fill, width=2)
                    
                else:
                    # SPEECH BUBBLE (Rounded, white background, black text, with tail)
                    box_fill = (255, 255, 255, 245)
                    text_fill = (0, 0, 0, 255)
                    outline_fill = (0, 0, 0, 255)
                    shadow_fill = (0, 0, 0, 100)
                    
                    # Draw drop shadow
                    draw.rounded_rectangle([x + 4, y + 4, x + bubble_width + 4, y + bubble_height + 4], radius=15, fill=shadow_fill)
                    
                    # Draw Bubble
                    draw.rounded_rectangle([x, y, x + bubble_width, y + bubble_height], radius=15, fill=box_fill, outline=outline_fill, width=3)
                    
                    # Draw Tail (Dynamic based on position)
                    tail_base_x = x + (bubble_width // 2)
                    if pos.startswith('top'):
                        # Tail points down
                        tail_tip_y = y + bubble_height + 30
                        tail_polygon = [(tail_base_x - 10, y + bubble_height - 3), (tail_base_x + 10, y + bubble_height - 3), (tail_base_x - 15, tail_tip_y)]
                        draw.polygon(tail_polygon, fill=box_fill, outline=outline_fill)
                        draw.line([(tail_base_x - 9, y + bubble_height - 3), (tail_base_x + 9, y + bubble_height - 3)], fill=box_fill, width=4)
                    else:
                        # Tail points up
                        tail_tip_y = y - 30
                        tail_polygon = [(tail_base_x - 10, y + 3), (tail_base_x + 10, y + 3), (tail_base_x + 15, tail_tip_y)]
                        draw.polygon(tail_polygon, fill=box_fill, outline=outline_fill)
                        draw.line([(tail_base_x - 9, y + 3), (tail_base_x + 9, y + 3)], fill=box_fill, width=4)
                
                # Draw Text
                text_y = y + padding_y
                for line in lines:
                    line_w = draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]
                    # Center text
                    line_x = x + (bubble_width - line_w) // 2
                    draw.text((line_x, text_y), line, font=font, fill=text_fill)
                    text_y += line_height + 2
                
            # Combine the overlay with the original image
            final_img = Image.alpha_composite(img, overlay).convert("RGB")
            final_img.save(output_path)
            return output_path
            
        except Exception as e:
            print(f"Error drawing speech bubble: {e}")
            return image_path

    def create_comic_page(self, image_paths: list, output_path: str):
        """Stitches multiple panels into a grid-based comic page layout."""
        if not image_paths:
            return None
            
        try:
            images = [Image.open(p) for p in image_paths]
            
            num_images = len(images)
            cols = 2 if num_images > 2 else 1
            rows = (num_images + cols - 1) // cols
            
            panel_width = images[0].width
            panel_height = images[0].height
            
            padding = 15 # Manga usually has tight gutters
            
            page_width = (panel_width * cols) + (padding * (cols + 1))
            page_height = (panel_height * rows) + (padding * (rows + 1))
            
            # Black background is standard for modern webtoons/dark fantasy manga
            page = Image.new("RGB", (page_width, page_height), "black")
            
            for i, img in enumerate(images):
                row = i // cols
                col = i % cols
                
                x = padding + (col * (panel_width + padding))
                y = padding + (row * (panel_height + padding))
                
                img = img.resize((panel_width, panel_height))
                
                # White border for panels on a black background
                bordered_img = Image.new("RGB", (panel_width + 6, panel_height + 6), "white")
                bordered_img.paste(img, (3, 3))
                
                page.paste(bordered_img, (x-3, y-3))
                
            page.save(output_path)
            return output_path
            
        except Exception as e:
            print(f"Error creating comic page: {e}")
            return None
