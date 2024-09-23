# ------------------------------------------------------------------------------------
# 번  호: 1
# 함수명: utils_nn_config_fn
# 입력값: model(은닉노드 모델)
# 설  명: keras 모델에서 각 층별로 정보를 추출
# ------------------------------------------------------------------------------------

# Towards DataScience에서 소스 가져오기
# 경로: https://towardsdatascience.com/deep-learning-with-python-neural-networks-complete-tutorial-6b53c0b06af0

def utils_nn_config_fn(model):
    lst_layers = []
    if 'Sequential' in str(model): #-> Sequential doesn't show the input layer
        layer = model.layers[0]
        lst_layers.append({'name':'input', 'in':int(layer.input.shape[-1]), 'neurons':0, 
                           'out':int(layer.input.shape[-1]), 'activation':None,
                           'params':0, 'bias':0})
    for layer in model.layers:
        print(layer.name)
        try:
            dic_layer = {'name':layer.name, 'in':int(layer.input.shape[-1]), 'neurons':layer.units, 
                         'out':int(layer.output.shape[-1]), 'activation':layer.get_config()['activation'],
                         'params':layer.get_weights()[0], 'bias':layer.get_weights()[1]}
        except:
            dic_layer = {'name':layer.name, 'in':int(layer.input.shape[-1]), 'neurons':0, 
                         'out':int(layer.output.shape[-1]), 'activation':None,
                         'params':0, 'bias':0}

        lst_layers.append(dic_layer)
    return lst_layers


# ------------------------------------------------------------------------------------
# 번  호: 2
# 함수명: visualize_nn_fn
# 입력값: model(은닉노드 모델), ...
# 설  명: utils_nn_config_fn()에서 추출한 정보를 바탕으로 신경망(Neural Network) 구조 시각화
# ------------------------------------------------------------------------------------

# Towards DataScience에서 소스 가져오기
# 경로: https://towardsdatascience.com/deep-learning-with-python-neural-networks-complete-tutorial-6b53c0b06af0

import matplotlib.pyplot as plt

# Plot the structure of a keras neural network.
def visualize_nn_fn(model, description = False, figsize = (10, 8), alpha = None, linewidth = None, col_names = []):
    ## get layers info
    lst_layers = utils_nn_config_fn(model)
    layer_sizes = [layer['out'] for layer in lst_layers]
    
    ## fig setup
    fig = plt.figure(figsize = figsize)
    ax = fig.gca()
    ax.set(title=model.name)
    ax.axis('off')
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    x_space = (right-left) / float(len(layer_sizes)-1)
    y_space = (top-bottom) / float(max(layer_sizes))
    p = 0.025
    
    ## nodes
    for i,n in enumerate(layer_sizes):
        top_on_layer = y_space*(n-1)/2.0 + (top+bottom)/2.0
        layer = lst_layers[i]
        color = 'green' if i in [0, len(layer_sizes)-1] else 'blue'
        color = 'red' if (layer['neurons'] == 0) and (i > 0) else color
        
        ### add description
        if (description is True):
            d = i if i == 0 else i-0.5
            if layer['activation'] is None:
                plt.text(x = left + d * x_space, y = top, fontsize = 10, color = color, s = layer['name'].upper())
            else:
                plt.text(x = left + d * x_space, y = top, fontsize = 10, color = color, s = layer['name'].upper())
                plt.text(x = left + d * x_space, y = top-p, fontsize = 10, color = color, s = layer['activation']+' (')
                plt.text(x = left + d * x_space, y = top-2*p, fontsize = 10, color = color, s = 'Σ'+str(layer['in'])+'[X*w]+b')
                out = ' Y'  if i == len(layer_sizes)-1 else ' out'
                plt.text(x = left + d * x_space, y = top-3*p, fontsize = 10, color = color, s =") = "+str(layer['neurons'])+out)
        
        ### circles
        for m in range(n):
            color = 'limegreen' if color == 'green' else color
            circle = plt.Circle(xy = (left + i*x_space, top_on_layer-m*y_space-4*p), radius = y_space/4.0, color = color, ec = 'k', zorder = 4)
            ax.add_artist(circle)
            
            ### add text
            if i == 0:
                # plt.text(x=left-4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$X_{'+str(m+1)+'}$')
                plt.text(x = left-4*p, y = top_on_layer-m*y_space-4*p, fontsize = 10, s = r'$'+col_names[m]+'$')
            elif i == len(layer_sizes)-1:
                plt.text(x = right + 4*p, y = top_on_layer-m*y_space-4*p, fontsize = 10, s = r'cum_runningtime')
            else:
                plt.text(x = left + i*x_space + p, y = top_on_layer-m*y_space + (y_space/8.+0.01*y_space)-4*p, fontsize = 10, s = r'$H_{'+str(m+1)+'}$')
    
    ## links
    for i, (n_a, n_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer = lst_layers[i + 1]
        color = 'green' if i == len(layer_sizes)-2 else 'blue'
        color = 'red' if layer['neurons'] == 0 else color
        layer_top_a = y_space*(n_a-1)/2. + (top+bottom)/2. -4*p
        layer_top_b = y_space*(n_b-1)/2. + (top+bottom)/2. -4*p
        for m in range(n_a):
            for o in range(n_b):
                line = plt.Line2D([i*x_space + left, (i + 1)*x_space + left], 
                                  [layer_top_a-m*y_space, layer_top_b-o*y_space], 
                                  c = color, alpha = alpha, linewidth = linewidth)
                if layer['activation'] is None:
                    if o == m:
                        ax.add_artist(line)
                else:
                    ax.add_artist(line)
    plt.show()