import os
# print(os.getcwd())
# os.chdir('./Machine Learning/Colab/Image Captioning')
# caption_file_path = './Flickr8k.token.txt'
train_file_path = './Flickr_8k.trainImages.txt'

def loadCaptions(file_path):
    capset = {}
    with open(file_path) as captions:
        c = captions.readline()
        while c:
            try:
                [file, cap] = c.split('\t')
                [file, num] = file.split('#')
                if capset.get(file):
                    capset[file].append(cap)
                else:
                    capset[file] = [cap]
                c = captions.readline()
            except:
                continue
    # print(len(capset.keys()))
    return capset

def getImages(file_path):
    img_list = []
    with open(file_path) as images:
        imgs = images.read().split('\n')
        for i in range(len(imgs)):
            try:
                file = imgs[i].split('.')
                if len(file)>0:
                    img_list.append(imgs[i])
            except:
                continue
    #
    return img_list

if __name__=='__main__':
    # capset = loadCaptions(caption_file_path)
    img_list = getImages(train_file_path)