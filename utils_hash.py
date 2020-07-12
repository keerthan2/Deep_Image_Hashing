from annoy import AnnoyIndex
import random

class SaveFeatures():
    features=None
    def __init__(self, m): 
        self.hook = m.avgpool.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output): 
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))
    def remove(self): 
        self.hook.remove()

def make_hash_database(model_ft,input_size,root_dir,TRANSFORM_IMG,device):
    sf = SaveFeatures(model_ft) ## Output before the last FC layer
    print(f"Creating hash for images from {root_dir}")
    data = ImageFolder(root=root_dir, transform=TRANSFORM_IMG)
    for ind in tqdm(range(len(data))):
        _= model_ft(data[ind][0].unsqueeze(0).to(device))
    print("Successfully created hash")
    return sf

def save_hash(feature_dict,f=512,hash_file_name = 'hash.ann'):
    t = AnnoyIndex(f, 'hamming') 
    k = 0
    for ip, vec in feature_dict.items():
        t.add_item(k, list(vec.flatten()))
        k += 1
    t.build(10000)
    t.save(hash_file_name)
    print(f"Saved hash at {hash_file_name}")

def find_closest_match(test_img_path,model_ft,TRANSFORM_IMG,device,f=512,n_items = 9):
    sf = SaveFeatures(model_ft)
    img = Image.open(test_img_path)
    test_img = TRANSFORM_IMG(img).unsqueeze(0).to(device)
    
    u = AnnoyIndex(f, 'hamming')
    u.load('hash.ann') 

    output = model_ft(test_img).cpu().detach().numpy().flatten()
    feature = sf.features[-1].flatten()

    columns = 3
    idxs = u.get_nns_by_vector(feature, n_items)
    rows = int(np.ceil(n_items/columns))
    fig=plt.figure(figsize=(max(2*rows,8), max(3*rows,8)))
    idx = 1
    for i in range(1, columns*rows +1):
        if i<n_items+1:
            img = Image.open(img_path[idxs[idx-1]])
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            idx += 1