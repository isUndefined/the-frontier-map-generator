"""No paid API requests: mock all image-edit responses."""
import base64
from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from PIL import Image
from providers.openai_edit import create_padded_inputs, get_dimensions, restore_original_fixed_pixels, generate

class OpenAIEditTests(unittest.TestCase):
    def test_native_resolution_and_mask(self):
        img = Image.new("RGB", (1672,941), (15,25,35))
        mask = Image.new("L", img.size, 255)
        mask.putpixel((0,0),0)
        canvas, api_mask = create_padded_inputs(img, mask)
        self.assertEqual(canvas.size,(1680,944))
        self.assertEqual(canvas.getpixel((0,0)),(15,25,35,255))
        self.assertEqual(api_mask.getpixel((0,0))[3],255)
        self.assertEqual(api_mask.getpixel((1,0))[3],0)

    def test_pixel_restore_and_dimension_rejection(self):
        img=Image.new("RGB",(800,832),(20,30,40))
        mask=Image.new("L",img.size,255)
        mask.putpixel((10,10),0)
        output=restore_original_fixed_pixels(img,Image.new("RGB",img.size,(200,200,200)),mask)
        self.assertEqual(output.getpixel((10,10)),(20,30,40,255))
        self.assertEqual(output.getpixel((11,10)),(200,200,200,255))
        with self.assertRaisesRegex(ValueError,"Refusing to resize"):
            restore_original_fixed_pixels(img,Image.new("RGB",(1024,1024)),mask)

    def test_unsupported_size_rejected(self):
        with self.assertRaises(ValueError):
            get_dimensions(10,10)

    def test_mocked_api_no_resize(self):
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            original=Image.new("RGB",(800,832),(7,8,9))
            mask=Image.new("L",original.size,255)
            mask.putpixel((0,0),0)
            original.save(temp/"canvas.png")
            mask.save(temp/"mask.png")
            (temp/"prompt.txt").write_text("Continue forest")
            data=BytesIO()
            Image.new("RGB",original.size,(100,110,120)).save(data,format="PNG")
            client=Mock()
            client.images.edit.return_value=Mock(data=[Mock(b64_json=base64.b64encode(data.getvalue()).decode())])
            generate(temp/"canvas.png",temp/"mask.png",temp/"prompt.txt",temp/"generated.png",client=client)
            self.assertEqual(client.images.edit.call_args.kwargs["size"],"800x832")
            with Image.open(temp/"generated.png") as result:
                self.assertEqual(result.size,(800,832))
                self.assertEqual(result.getpixel((0,0)),(7,8,9))
                self.assertEqual(result.getpixel((1,0)),(100,110,120))

if __name__=="__main__":
    unittest.main()
