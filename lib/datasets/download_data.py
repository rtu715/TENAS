import os 
import boto3

def download_from_s3(s3_bucket, task, download_dir):
    s3 = boto3.client("s3")

    if task == 'scifar100':
        data_files = ["s2_cifar100.gz"]
        s3_folder = 'spherical'

    elif task == 'ninapro':
        data_files = ['ninapro_train.npy', 'ninapro_val.npy', 'ninapro_test.npy',
                      'label_train.npy', 'label_val.npy', 'label_test.npy']
        s3_folder = 'ninapro'

    elif task =='cifar10' or task =='cifar100': 
        return

    elif task == 'audio':
        data_files = ['audio.zip']
        s3_folder = 'audio'

    elif task == 'darcyflow':
        data_files = ["piececonst_r421_N1024_smooth1.mat", "piececonst_r421_N1024_smooth2.mat"]
        s3_folder = None

    else:
        raise NotImplementedError

    for data_file in data_files:
        filepath = os.path.join(download_dir, data_file)
        if s3_folder is not None:
            s3_path = os.path.join(s3_folder, data_file)
        else:
            s3_path = data_file
        if not os.path.exists(filepath):
            s3.download_file(s3_bucket, s3_path, filepath)
        
    #extract zip if audio
    if task == 'audio' and not os.path.exists(os.path.join(download_dir, 'data')):
        #raise ValueError('Check dir')
        os.mkdir(os.path.join(download_dir,'data'))
        import zipfile
        with zipfile.ZipFile(os.path.join(download_dir, 'audio.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(download_dir, 'data'))

    return

def download_protein_folder(bucket_name, local_dir=None):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix='protein'):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, 'protein'))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)