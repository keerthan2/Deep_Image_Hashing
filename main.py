import warnings
warnings.filterwarnings('ignore')

from utils_train_test import *
from utils_hash import *
from model import initialize_model

pretrained_model_ckpt = "./pretrained_checkpoints" 
if not os.path.exists(pretrained_model_ckpt):
  os.mkdir(pretrained_model_ckpt)
os.environ['TORCH_HOME'] = pretrained_model_ckpt

model_name = 'resnet'
log_dir = './Checkpoints/'
root_dir = './grocery_data' ## Change this to your training dataset path
test_img_path = 'test_imgs/pizza2.jpg' ## Change this to your test image path

num_closest_matches = 9
num_epochs = 15
BATCH_SIZE = 64
num_classes = 10

isTrain = True
isSaveHash = True
feature_extract = False
train_val_split = 0.3
load_model = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_name = log_dir+model_name+'.pth'
if not os.path.exists(log_dir):
  os.mkdir(log_dir)

if load_model:
  print(f"Loading model....")
  model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
  model_ft.load_state_dict(torch.load(save_name,map_location=device))
else:
  print(f"Instantiating {model_name} model")
  model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

model_ft = model_ft.to(device)

TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )
        ])


print(f"Loading images from {root_dir}")
data = ImageFolder(root=root_dir, transform=TRANSFORM_IMG)
num_val = int(train_val_split*len(data))
num_train = len(data) - num_val
train_set, val_set = torch.utils.data.random_split(data, [num_train, num_val])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader  = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

if isTrain:
    print(f"Starting to train on {device}")
    params_to_update = model_ft.parameters()
    optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9, weight_decay=4e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, threshold=1e-3, eps=1e-7)

    best_acc = 0
    for epoch in range(num_epochs): 
        model_ft = train(model_ft,train_loader,criterion,optimizer_ft,epoch,device)
        val_loss, best_acc = evaluate(model_ft,val_loader,criterion,best_acc,save_name,device)
        scheduler.step(val_loss)
    print(f"Best Validation Accuracy: {best_acc}%")
    model_ft.load_state_dict(torch.load(save_name,map_location=device))

model_ft = model_ft.to(device).eval()

if isSaveHash:
    sf = make_hash_database(model_ft,input_size,root_dir,TRANSFORM_IMG,device)
    img_path = [s[0] for s in data.samples]
    feature_dict = dict(zip(img_path,sf.features))
    save_hash(feature_dict,f=512,hash_file_name = 'hash.ann')

find_closest_match(test_img_path,model_ft,TRANSFORM_IMG,device,f=512,n_items = num_closest_matches)




