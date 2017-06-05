from os.path import join

## image and csv paths

imgspath = '../simdata'
trainpath = '18kforward'
trainpath2 = '9krecoveries'
testpath = 'testdata/5kforward'
csvfile = 'driving_log.csv'
datapaths = [
    (join(imgspath, trainpath, csvfile), join(imgspath, trainpath, 'IMG')),
    (join(imgspath, trainpath2,csvfile), join(imgspath, trainpath2,'IMG')),
]
testpaths = (join(testpath, csvfile), join(testpath, 'IMG'))
