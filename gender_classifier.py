import argparse
import tensorflow as tf
from src.adience_model import AdienceModel
from src.adience_data import AdienceData
from src.vgg_face import VggFace

def main():
    parser = setup_parsers()
    args = parser.parse_args()
    args.func(args)

def train_model(args):
    print('training the model...')
    
    #create vgg-face model and load weights
    vgg_face = VggFace()
    vgg_face.load_weights(args.vggface_weights)

    #create data and datagenerators for adience data
    train_data = AdienceData(args.img_directory, "train", memory = args.memory, batch_size = args.train_batch_size)
    valid_data = AdienceData(args.img_directory, "valid", memory = args.memory, batch_size = args.validation_batch_size)

    #create adience gender classifier model and compile
    adience_model = AdienceModel(vgg_face.model)
    adience_model.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    #train the model
    history = adience_model.model.fit(train_data.datagen, validation_data = valid_data.datagen, epochs = args.epochs)
    
    #save the model
    adience_model.model.save(args.output, save_format = "h5")
    
    print('model trained and stored at ' + args.output)
    
def classify_image(args):
    #load trained model
    adience_model = AdienceModel()
    adience_model.load_model(args.model)
    
    #classify input image
    output = adience_model.classify_single_image(args.image)
    
    print('****')
    print(f'output class is {output[1]}. (sigmoid value={output[0]})')
    print('****')     
    
def evaluate_model(args):
    #create data and datagenerators for adience data
    train_data = AdienceData(args.img_directory, "train", memory = args.memory, batch_size = args.train_batch_size)
    valid_data = AdienceData(args.img_directory, "valid", memory = args.memory, batch_size = args.validation_batch_size)
    
    #load trained model
    adience_model = AdienceModel()
    adience_model.load_model(args.model)
    
    #evaluating
    print('evaluating model on validation set...')
    adience_model.model.evaluate(valid_data.datagen)
    
    print('evaluating model on training set...')
    adience_model.model.evaluate(train_data.datagen)

def setup_parsers():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help = 'train, evaluate, or classify')
    train_parser = subparsers.add_parser("train")
    classify_parser = subparsers.add_parser("classify")
    eval_parser = subparsers.add_parser("evaluate")
    train_parser = setup_train_parser(train_parser)
    eval_parser = setup_eval_parser(eval_parser)
    classify_parser = setup_classify_parser(classify_parser)
    parser.add_argument('-b1', '--train-batch-size', help = 'batch size for training set', 
                        type = int, required = False, default = 32)
    parser.add_argument('-b2', '--validation-batch-size', help = 'batch size for validation set',
                        type = int, required = False, default = 32)
    parser.add_argument('-m', '--memory', choices = ['low', 'high'], default = "low", required = False,
                        help = 'whether to use high-memory setting (load all images in memory)')
    return parser
    
def setup_train_parser(parser):
    parser.add_argument('-w', '--vggface-weights', required = True, help = 'path to the t7 file containing ' +
                                                          'vgg-face weights (downloaded torch directory)')
    parser.add_argument('-i', '--img-directory', help = 'path to adience images directory containing ' +
                        'aligned and valid subdirectoried.', required = True)
    parser.add_argument('-o', '--output', required = True, help = 'where to save the trained model')
    parser.add_argument('-e', '--epochs', required = False, default = 1, help = 'epochs to train', type = int)
    parser.set_defaults(func=train_model)
    return parser
    
def setup_classify_parser(parser):
    parser.add_argument('-m', '--model', required = True, help = 'path to the trained and stored model')
    parser.add_argument('-i', '--image', help = 'image to classify', required = True)
    parser.set_defaults(func=classify_image)
    return parser
                        
def setup_eval_parser(parser):
    parser.add_argument('-m', '--model', required = True, help = 'path to the trained and stored model')
    parser.add_argument('-i', '--img-directory', help = 'path to adience images directory containing ' +
                        'aligned and valid subdirectoried.', required = True)
    parser.set_defaults(func=evaluate_model)
    return parser

if __name__ == "__main__":
    main()
