from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from PIL import Image
import io
import fitz  # PyMuPDF
import uuid
import os
import base64

def init_gemini(api_key):
    """Initialize Gemini API with the provided key."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.1
    )

def describe_image(model, image_data, prompt="What's in this image? Provide full detail as possible."):
    """Get a description of an image using Gemini Vision."""
    try:
        # Convert bytes to PIL Image if needed
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data
        
        print(f"Processing image: {image.size} {image.mode}")
        
        # Convert PIL image to base64 for LangChain
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        image_url = f"data:image/png;base64,{img_str}"
        
        # Create message with image and text
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt,
                },
                {"type": "image_url", "image_url": image_url},
            ]
        )
        
        # Generate description
        response = model.invoke([message])
        print(f"Generated description: {response.content[:100]}...")
        return response.content
        
    except Exception as e:
        print(f"Error describing image: {e}")
        return None

def extract_images_from_pdf(pdf_path):
    """Extract images from PDF using PyMuPDF with simplified processing."""
    try:
        doc = fitz.open(pdf_path)
        images = []
        
        print(f"Extracting images from PDF: {pdf_path}")
        print(f"PDF has {len(doc)} pages")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            
            print(f"Page {page_num + 1} has {len(image_list)} images")
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    print(f"Image {img_index} on page {page_num + 1}: {pix.width}x{pix.height}, colorspace: {pix.colorspace.n}, alpha: {pix.alpha}")
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        images.append({
                            "page": page_num + 1,
                            "image_data": img_data,
                            "image_id": f"page_{page_num+1}_img_{img_index}_{uuid.uuid4().hex[:8]}"
                        })
                        print(f"Successfully extracted image {img_index} from page {page_num + 1}")
                    else:
                        print(f"Skipping image {img_index} from page {page_num + 1} (unsupported format)")
                    
                    pix = None  # Free memory
                except Exception as e:
                    print(f"Error extracting image {img_index} from page {page_num}: {e}")
                    continue
        
        doc.close()
        print(f"Total images extracted: {len(images)}")
        return images
    except Exception as e:
        print(f"Error extracting images from PDF: {str(e)}")
        return []

def process_pdf_images(pdf_path, model):
    """Process all images in a PDF and return their descriptions."""
    print(f"Starting image processing for: {pdf_path}")
    images = extract_images_from_pdf(pdf_path)
    results = []
    
    print(f"Processing {len(images)} images with Gemini Vision")
    
    for i, img in enumerate(images):
        print(f"Processing image {i+1}/{len(images)} from page {img['page']}")
        description = describe_image(model, img["image_data"])
        if description:
            results.append({
                "page": img["page"],
                "image_id": img["image_id"],
                "description": description
            })
            print(f"Successfully processed image {i+1}")
        else:
            print(f"Failed to process image {i+1}")
    
    print(f"Final results: {len(results)} image descriptions generated")
    return results 