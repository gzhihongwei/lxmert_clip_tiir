import os
import random
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((225, 225)),
    transforms.ToTensor()
])

def get_loader(batch_size, root, ann_file="annotations/captions_train2014.json", 
               transform=None, prob_unaligned=0.9, num_workers=0):
    dataset = CoCoText2Img(root=root, 
                           ann_file=ann_file, 
                           transform=transform,
                           prob_unaligned=prob_unaligned)
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    
    return data_loader


class CoCoText2Img(data.Dataset):
    def __init__(self, root, ann_file, transform, prob_unaligned):
        from pycocotools.coco import COCO
        
        # Store the transform
        self.transform = transform
        # Store the image root
        self.root = root
        # Create a COCO instance
        self.coco = COCO(ann_file)
        # All of the caption ids
        self.ids = list(sorted(self.coco.anns.keys()))
        # Probability of returning an unaligned image from caption
        assert 0 <= prob_unaligned <= 1, "prob_unaligned must be a probability" 
        self.prob_unaligned = prob_unaligned
        
    def _get_caption_image_id(self, ann_id):
        return self.coco.loadAnns(ann_id)[0]["image_id"]
    
    def _load_caption(self, ann_id):
        return self.coco.loadAnns(ann_id)[0]["caption"]
    
    def _load_image(self, image_id):
        path = self.coco.loadImgs(image_id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")
    
    def _load_unaligned_image(self, true_image_id):
        candidate_image_ids = list({anns["image_id"] for anns in self.coco.anns.values() if anns["image_id"] != true_image_id})
        unaligned_image_id = random.choice(candidate_image_ids)
        return self._load_image(unaligned_image_id)
        
    
    def __getitem__(self, index):
        id = self.ids[index]
        caption = self._load_caption(id)
        image_id = self._get_caption_image_id(id)
        image = self._load_image(image_id)
        aligned = 1
        
        if random.random() < self.prob_unaligned:
            aligned = 0
            image = self._load_unaligned_image(id)
            
        if self.transform is not None:
            image = self.transform(image)
            
        return caption, image, aligned
    
    def __len__(self):
        return len(self.ids)
        
        
if __name__ == "__main__":
    os.chdir(os.path.join("..", "..", "datasets", "coco"))
    train_loader = get_loader(batch_size=32,
                              root=os.path.join("images", "train2014"),
                              ann_file=os.path.join("annotations", "captions_train2014.json"),
                              transform=IMAGE_TRANSFORM)
    print(next(iter(train_loader)))
    
    
  
