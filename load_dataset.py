import os
# print(os.getcwd())
# os.chdir('./Machine Learning/Colab/Image Captioning')
# caption_file_path = './Flickr8k.token.txt'
train_file_path = './Flickr_8k.trainImages.txt'

def loadCaptions(file_path):
    capdict = {}
    with open(file_path) as captions:
        c = captions.readline()
        while c:
            try:
                [file, cap] = c.split('\t')
                [file, num] = file.split('#')
                if capdict.get(file):
                    capdict[file].append(cap)
                else:
                    capdict[file] = [cap]
                c = captions.readline()
            except:
                continue
    # print(len(capdict.keys()))
    return capdict

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
    # capdict = loadCaptions(caption_file_path)
    img_list = getImages(train_file_path)