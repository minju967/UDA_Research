from build_model   import Encoder, MLP

class Trainer:
    def __init__(self, args, train_loader, test_loader) -> None:
        self.args = args
        self.train_dataset = train_loader
        self.test_dataset  = test_loader
    
    def set_networks(self):
        self.Encoder = Encoder()
        self.MLP     = MLP(64, 64)

    def set_lossfunction(self):
        self.loss = None

    def get_batch(self, batch_data_iter):
        batch_data = dict()
        for dset in self.args.datasets:
            try:
                batch_data[dset] = next(batch_data_iter[dset])
            except StopIteration:
                batch_data_iter[dset] = iter(self.train_dataset[dset])
                batch_data[dset] = next(batch_data_iter[dset])
        
        return batch_data

    def extract_feature(self, imgs1, imgs2):
        features, global_vec, domain_vec = dict(), dict(), dict()
        for dset in self.args.datasets:
            fea_map1, g_vec  = self.Encoder(imgs1[dset])
            fea_map2, d_vec  = self.Encoder(imgs2[dset])

            features[dset] = fea_map2
            global_vec[dset] = g_vec
            domain_vec[dset] = d_vec
        
            ## multiplication(batch, 64) or concat dim=1(batch, 128)
            multi_vec = g_vec * d_vec
            multi_vec = multi_vec.view(multi_vec.size(0), -1)
            content_vec = self.MLP(multi_vec)       

        return features, global_vec, domain_vec, content_vec
    
    def train(self):
        self.set_networks()
        self.set_lossfunction()
        batch_data_iter = dict()
        for dset in self.args.datasets:
            batch_data_iter[dset] = iter(self.train_dataset[dset])

        for i in range(self.args.iter):     # not epoch using iteration
            batch_data = self.get_batch(batch_data_iter)
            imgs_1, imgs_2, labels =  dict(), dict(), dict()
            min_batch = self.args.batch
            for dset in self.args.datasets:
                imgs_1[dset], imgs_2[dset], labels[dset] = batch_data[dset]
                # imgs[dset], labels[dset] = imgs[dset].cuda(), labels[dset].cuda()
                if imgs_1[dset].size(0) < min_batch:
                    min_batch = imgs_1[dset].size(0)
            if min_batch < self.args.batch:
                for dset in self.args.datasets:
                    imgs_1[dset], labels[dset] = imgs_1[dset][:min_batch], labels[dset][:min_batch]

            feature_map, global_vec, domain_vec, content_vec = self.extract_feature(imgs_1, imgs_2)
            print(f'content vector size: {content_vec.size()}')

            ## loss function define
        