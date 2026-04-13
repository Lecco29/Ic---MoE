import sys
#Please, change the following path to where convnet2 can be located
# Download convnet2 from: https://github.com/jmsaavedrar/convnet2
# Then update the path below to point to your convnet2 directory
# Example for Windows: sys.path.append(r"C:\path\to\convnet2")
# Example for Linux/Mac: sys.path.append("/path/to/convnet2")
# no meu caso, o convnet2 está na pasta C:\Users\leo\Desktop\IC Arlete\convnet2
#esse script foi modificado para gerar a acuracia e testar todos os blocos usando todo o dataset
sys.path.append(r"C:\Users\leo\Desktop\IC Arlete\convnet2")
import tensorflow as tf
import datasets.data as data
import utils.configuration as conf
import utils.imgproc as imgproc
import skimage.io as io
import skimage.transform as trans
import os
import argparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

class SSearch :
    def __init__(self, config_file, model_name):
        
        self.configuration = conf.ConfigurationFile(config_file, model_name)
        #defiing input_shape                    
        self.input_shape =  (self.configuration.get_image_height(), 
                             self.configuration.get_image_width(),
                             self.configuration.get_number_of_channels())                       
        #loading the model
        model = tf.keras.applications.ResNet50(include_top=True, 
                                               weights='imagenet', 
                                               input_tensor=None, 
                                               input_shape =self.input_shape, 
                                               pooling=None, 
                                               classes=1000)
        model.summary()
        #redefining the model to get the hidden output
        color_layer =  'conv2_block3_out'
        texture_layer =  'conv4_block6_out'
        self.output_layer_name = color_layer
        output = model.get_layer(self.output_layer_name).output
        output = tf.keras.layers.GlobalAveragePooling2D()(output)                
        self.sim_model = tf.keras.Model(model.input, output)        
        self.sim_model.summary()
        
        # fixando todos os blocos para avaliação
        self.blocks = {
            'Block1': 'conv1_relu',
            'Block2': 'conv2_block3_out',
            'Block3': 'conv3_block4_out',
            'Block4': 'conv4_block6_out',
            'Block5': 'conv5_block3_out'
        }
        
        # Criar modelos para cada bloco
        self.block_models = {}
        for block_name, layer_name in self.blocks.items():
            try:
                output = model.get_layer(layer_name).output
                output = tf.keras.layers.GlobalAveragePooling2D()(output)
                self.block_models[block_name] = tf.keras.Model(model.input, output)
            except Exception as e:
                print(f'[ERRO] Erro ao criar {block_name} ({layer_name}): {e}')
        #defining image processing function
        self.process_fun =  imgproc.process_image_visual_attribute
        #loading catalog
        self.ssearch_dir = os.path.join(self.configuration.get_data_dir(), 'ssearch')
        catalog_file = os.path.join(self.ssearch_dir, 'catalog.txt')        
        assert os.path.exists(catalog_file), '{} does not exist'.format(catalog_file)
        print('loading catalog ...')
        self.load_catalog(catalog_file)
        print('loading catalog ok ...')
        self.enable_search = False
        
        self._extract_labels()        
        
    #read_image
    def read_image(self, filename):        
        im = self.process_fun(data.read_image(filename, self.input_shape[2]), (self.input_shape[0], self.input_shape[1]))        
        #for resnet
        im = tf.keras.applications.resnet50.preprocess_input(im)    
        return im
    
    def load_features(self):
        fvs_file = os.path.join(self.ssearch_dir, "features.np")                        
        fshape_file = os.path.join(self.ssearch_dir, "features_shape.np")
        features_shape = np.fromfile(fshape_file, dtype = np.int32)
        self.features = np.fromfile(fvs_file, dtype = np.float32)
        self.features = np.reshape(self.features, features_shape)
        self.enable_search = True
        print('features loaded ok')
        
    def load_catalog(self, catalog):
        with open(catalog, encoding='utf-8') as f_in :
            self.filenames = [filename.strip() for filename in f_in if filename.strip()]
        self.data_size = len(self.filenames)
    
    # essa função vai extrair as labels de cores e texturas dos nomes dos arquivos
    def _extract_labels(self):
        self.labels_color = []
        self.labels_texture = []
        
        for filename in self.filenames:
            if 'color/' in filename:
                color = filename.split('color/')[1].split('/')[0]
                self.labels_color.append(color)
                self.labels_texture.append(None)
            elif 'texture/' in filename:
                texture = filename.split('texture/')[1].split('/')[0]
                self.labels_texture.append(texture)
                self.labels_color.append(None)
            else:
                self.labels_color.append(None)
                self.labels_texture.append(None)
        
        # Separando o s indices válidos
        self.color_indices = [i for i, label in enumerate(self.labels_color) if label is not None]
        self.texture_indices = [i for i, label in enumerate(self.labels_texture) if label is not None]
        
        print(f'Labels extraidos: {len(self.color_indices)} cores, {len(self.texture_indices)} texturas')    
            
    def get_filenames(self, idxs):
        return [self.filenames[i] for i in idxs]
        
    def compute_features(self, image, expand_dims = False):
        #image = image - self.mean_image
        if expand_dims :
            image = tf.expand_dims(image, 0)        
        fv = self.sim_model.predict(image)            
        return fv
    
    def normalize(self, data) :
        """
        unit normalization
        """
        norm = np.sqrt(np.sum(np.square(data), axis = 1))
        norm = np.expand_dims(norm, 0)  
        print(norm)      
        data = data / np.transpose(norm)
        return data
    
    def square_root_norm(self, data) :
        return self.normalize(np.sign(data)*np.sqrt(np.abs(data)))        
 
    def search(self, im_query, metric = 'l2', norm = 'None'):
        assert self.enable_search, 'search is not allowed'
        q_fv = self.compute_features(im_query, expand_dims = True)
        #it seems that Euclidean performs better than cosine
        if metric == 'l2' :
            data = self.features
            query =q_fv            
            if norm == 'square_root' :
                data = self.square_root_norm(data)
                query = self.square_root_norm(query)
            d = np.sqrt(np.sum(np.square(data - query[0]), axis = 1))
            idx_sorted = np.argsort(d)
            print(d[idx_sorted][:20])
        elif metric == 'cos' : 
            sim = np.matmul(self.normalize(self.features), np.transpose(self.normalize(q_fv)))
            sim = np.reshape(sim, (-1))            
            idx_sorted = np.argsort(-sim)
            print(sim[idx_sorted][:20])                
        return idx_sorted[:90]
        
                                
    def compute_features_from_catalog(self):
        n_batch = self.configuration.get_batch_size()        
        images = np.empty((self.data_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype = np.float32)
        for i, filename in enumerate(self.filenames) :
            if i % 1000 == 0:
                print('reading {}'.format(i))
                sys.stdout.flush()
            images[i, ] = self.read_image(filename)        
        n_iter = int(np.ceil(self.data_size / n_batch))
        result = []
        for i in range(n_iter) :
            print('iter {} / {}'.format(i, n_iter))  
            sys.stdout.flush()             
            batch = images[i*n_batch : min((i + 1) * n_batch, self.data_size), ]
            result.append(self.compute_features(batch))
        fvs = np.concatenate(result)    
        print('fvs {}'.format(fvs.shape))    
        fvs_file = os.path.join(self.ssearch_dir, "features.np")
        fshape_file = os.path.join(self.ssearch_dir, "features_shape.np")
        np.asarray(fvs.shape).astype(np.int32).tofile(fshape_file)       
        fvs.astype(np.float32).tofile(fvs_file)
        print('fvs saved at {}'.format(fvs_file))
        print('fshape saved at {}'.format(fshape_file))

    def draw_result(self, filenames):
        w = 1000
        h = 1000
        w_i = int(w / 10)
        h_i = int(h / 10)
        image_r = np.zeros((w,h,3), dtype = np.uint8) + 255
        x = 0
        y = 0
        for i, filename in enumerate(filenames) :
            pos = (i * w_i)
            x = pos % w
            y = int(np.floor(pos / w)) * h_i
            image = data.read_image(filename, 3)            
            image = imgproc.toUINT8(trans.resize(image, (h_i,w_i)))
            image_r[y:y+h_i, x : x +  w_i, :] = image              
        return image_r
    
    def compute_all_blocks_features(self):
        """Computa features de todas as imagens para todos os blocos"""
        print("\n" + "="*70)
        print("EXTRAINDO FEATURES DE TODOS OS BLOCOS")
        print("="*70)
        
        # Carregar todas as imagens
        print("\nCarregando imagens...")
        images = np.empty((self.data_size, self.input_shape[0], 
                          self.input_shape[1], self.input_shape[2]), dtype=np.float32)
        
        for i, filename in enumerate(self.filenames):
            if i % 500 == 0:
                print(f'  Processando imagem {i}/{self.data_size}')
            images[i] = self.read_image(filename)
        
        # Extrair features de cada bloco
        all_features = {}
        for block_name in self.blocks.keys():
            print(f'\nExtraindo features do {block_name}...')
            model = self.block_models[block_name]
            n_batch = self.configuration.get_batch_size()
            n_iter = int(np.ceil(self.data_size / n_batch))
            
            result = []
            for i in range(n_iter):
                if i % 2 == 0:
                    print(f'  Batch {i+1}/{n_iter}')
                batch = images[i*n_batch : min((i + 1) * n_batch, self.data_size)]
                features = model.predict(batch, verbose=0)
                result.append(features)
            
            all_features[block_name] = np.concatenate(result)
            print(f'  Shape: {all_features[block_name].shape}')
        
        return all_features
    
    # essa função vai salvar as features de todos os blocos
    def save_block_features(self, all_features):
        print("\nSalvando...")
        for block_name, features in all_features.items():
            fvs_file = os.path.join(self.ssearch_dir, f"features_{block_name}.np")
            fshape_file = os.path.join(self.ssearch_dir, f"features_{block_name}_shape.np")
            
            np.asarray(features.shape).astype(np.int32).tofile(fshape_file)
            features.astype(np.float32).tofile(fvs_file)
            print(f'  {block_name} salvo: {fvs_file}')
    
    # essa função vai carregar as features de um bloco específico
    def load_block_features(self, block_name):
        fvs_file = os.path.join(self.ssearch_dir, f"features_{block_name}.np")
        fshape_file = os.path.join(self.ssearch_dir, f"features_{block_name}_shape.np")
        
        if not os.path.exists(fvs_file):
            return None
        
        features_shape = np.fromfile(fshape_file, dtype=np.int32)
        features = np.fromfile(fvs_file, dtype=np.float32)
        features = np.reshape(features, features_shape)
        return features
    
    def evaluate_block_knn(self, features, block_name, attribute_type='color', k=5):
        """Avalia um bloco usando kNN com validação cruzada"""
        print(f'\nAvaliando {block_name} para {attribute_type}...')
        
        # aqui vai filtrar as features e labels validas
        if attribute_type == 'color':
            valid_indices = self.color_indices
            valid_labels = [self.labels_color[i] for i in valid_indices]
        else:  
            valid_indices = self.texture_indices
            valid_labels = [self.labels_texture[i] for i in valid_indices]
        
        if len(valid_indices) == 0:
            print(f'  Nenhuma imagem de {attribute_type} encontrada!')
            return 0.0, 0.0
        
        X = features[valid_indices]
        y = np.array(valid_labels)
        
        # aqui vai usar a validação cruzada com 5 folds
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Treinando o kNN com a metrica euclidiana
            knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            knn.fit(X_train, y_train)
            
            # previsao do kNN
            y_pred = knn.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
        
        # aqui vai calcular a acuracia media e o desvio padrao
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f'  Accuracy: {mean_acc:.3f} ± {std_acc:.3f}')
        return mean_acc, std_acc
    
    # aqui vai avaliar todos os blocos e gerar o relatório de acurácia
    def evaluate_all_blocks(self, all_features=None, k=5):
        print("usando o kNN para avaliar todos os blocos")
        
        # aqui vai carregar as features se não foram fornecidas
        if all_features is None:
            print("\nCarregando features salvas...")
            all_features = {}
            for block_name in self.blocks.keys():
                features = self.load_block_features(block_name)
                if features is not None:
                    all_features[block_name] = features
                    print(f'  {block_name} carregado: {features.shape}')
                else:
                    print(f'  [AVISO] Features de {block_name} nao encontradas!')
                    print(f'  Execute primeiro: compute_all_blocks')
                    return None
        
        # Avaliar cada bloco
        results = {}
        
        for block_name in self.blocks.keys():
            if block_name not in all_features:
                print(f'\n[AVISO] Pulando {block_name} - features nao disponiveis')
                continue
                
            results[block_name] = {}
            
            # cores
            acc_color, std_color = self.evaluate_block_knn(
                all_features[block_name], 
                block_name, 
                'color',
                k
            )
            results[block_name]['color'] = acc_color
            results[block_name]['color_std'] = std_color
            
            # texturas
            acc_texture, std_texture = self.evaluate_block_knn(
                all_features[block_name], 
                block_name, 
                'texture',
                k
            )
            results[block_name]['texture'] = acc_texture
            results[block_name]['texture_std'] = std_texture
        
        # Imprimi a acuracia de cada bloco
        print("Acuracia (kNN, k={}, 5-fold CV)".format(k))
        print(f"{'Block':<10} {'Color':<20} {'Texture':<20}")
        print("-"*50)
        
        for block_name in self.blocks.keys():
            if block_name in results:
                color_acc = results[block_name]['color']
                texture_acc = results[block_name]['texture']
                print(f"{block_name:<10} {color_acc:<20.3f} {texture_acc:<20.3f}")
        
        # Salvar todos os resultados
        results_file = os.path.join(self.ssearch_dir, "resultados_acuracia.txt")
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("Acuracia(kNN, k={}, 5-fold CV)\n".format(k))
            f.write(f"{'Block':<10} {'Color':<15} {'Texture':<15}\n")
            f.write("-"*40 + "\n")
            for block_name in self.blocks.keys():
                if block_name in results:
                    f.write(f"{block_name:<10} {results[block_name]['color']:<15.3f} {results[block_name]['texture']:<15.3f}\n")
        
        print(f"\n[OK] O resultado foi salvo no arquivo: {results_file}")
        
        return results
                    
#unit test        
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = "Similarity Search")        
    parser.add_argument("-config", type = str, help = "<str> configuration file", required = True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)                
    parser.add_argument("-mode", type=str, choices = ['search', 'compute', 'compute_all_blocks', 'evaluate_blocks'], help=" mode of operation", required = True)
    parser.add_argument("-list", type=str,  help=" list of image to process", required = False)
    parser.add_argument("-odir", type=str,  help=" output dir", required = False, default = '.')
    parser.add_argument("-k", type=int, help=" number of neighbors for kNN", required = False, default = 5)
    pargs = parser.parse_args()     
    configuration_file = pargs.config        
    ssearch = SSearch(pargs.config, pargs.name)
    metric = 'cos'
    norm = 'square_root'
    if pargs.mode == 'compute' :        
        ssearch.compute_features_from_catalog()
    elif pargs.mode == 'compute_all_blocks':
        # salva as features de todos os blocos
        all_features = ssearch.compute_all_blocks_features()
        ssearch.save_block_features(all_features)
        print("\n[OK] Features de todos os blocos computadas e salvas!")
    elif pargs.mode == 'evaluate_blocks':
        # usando o kNN para avaliar todos os blocos
        results = ssearch.evaluate_all_blocks(k=pargs.k)
        if results:
            print("\n[OK] Concluido")
    if pargs.mode == 'search' :
        ssearch.load_features()        
        if pargs.list is not None :
            with open(pargs.list) as f_list :
                filenames  = [ item.strip() for item in f_list]
            for fquery in filenames :
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query, metric)                
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)#           
                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_{}_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))                
        else :
            fquery = input('Query:')
            while fquery != 'quit' :
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query, metric)                
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)    
                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))
                fquery = input('Query:')
        
        
