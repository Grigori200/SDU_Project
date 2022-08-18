from models import Classifier


if __name__ == '__main__':
    model = Classifier.load_from_checkpoint('save/xgb/epoch22-step2162.ckpt').model
    for mod in model.modules():
        print(mod)
