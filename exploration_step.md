# Dataset Information

- Contains 60 000 images

# Sparse VS One Hot

Sparse is most of the better than One Hot. ( refers to the doc Sparse VS One hot)

# Linear

## Test all combinaisons of Activation / Optimizers with:
- learning rate : 0.0001
- batchs : 5000 / 1000
- epochs : 500
- Norm / no Norm
- Gray / Color
- loss : categorical_crossentropy


##### Note :  These configurations are not working with tensorboard because of Nan

- linear,sgd,categorical_crossentropy,500,5000_0.0001_True_False
- linear,sgd,categorical_crossentropy,500,1000_0.0001_True_False
- linear,sgd,categorical_crossentropy,500,1000,0.0001,False,False
- linear,sgd,categorical_crossentropy,500,5000,0.0001,False,False

=> the couple linear + Sgd is only working if Norm is set to True


#### Best 10 configs for Linear

| activation | optimizer | loss                     | epochs | batch-size | learning-rate | isGray | isNorm | last_loss | last_val_loss | last_accuracy | last_val_accuracy |
|------------|-----------|--------------------------|--------|------------|---------------|--------|--------|-----------|---------------|---------------|-------------------|
| linear     | adamax    | categorical_crossentropy | 500    | 5000       | 0.0001        | FAUX   | VRAI   | 2.05683   | 2.0691        | 0.18965       | 0.1941            |
| selu       | adam      | categorical_crossentropy | 500    | 1000       | 0.0001        | FAUX   | VRAI   | 2.02468   | 2.04076       | 0.19308       | 0.1928            |
| linear     | adamax    | categorical_crossentropy | 500    | 1000       | 0.0001        | FAUX   | VRAI   | 2.06132   | 2.06988       | 0.19488       | 0.1926            |
| linear     | adam      | categorical_crossentropy | 500    | 1000       | 0.0001        | FAUX   | VRAI   | 2.0378    | 2.05222       | 0.19435       | 0.1921            |
| elu        | rmsprop   | categorical_crossentropy | 500    | 1000       | 0.0001        | FAUX   | VRAI   | 2.07778   | 2.07607       | 0.1873        | 0.1892            |
| elu        | adamax    | categorical_crossentropy | 500    | 1000       | 0.0001        | FAUX   | VRAI   | 2.0679    | 2.08643       | 0.19135       | 0.1882            |
| linear     | sgd       | categorical_crossentropy | 500    | 1000       | 0.0001        | FAUX   | VRAI   | 2.09911   | 2.10564       | 0.1868        | 0.1881            |
| linear     | nadam     | categorical_crossentropy | 500    | 1000       | 0.0001        | FAUX   | VRAI   | 2.05672   | 2.0648        | 0.18955       | 0.1879            |
| softplus   | rmsprop   | categorical_crossentropy | 500    | 1000       | 0.0001        | FAUX   | VRAI   | 2.06619   | 2.08833       | 0.19200       | 0.1855            |
| linear     | nadam     | categorical_crossentropy | 500    | 5000       | 0.0001        | FAUX   | VRAI   | 2.09998   | 2.10575       | 0.18927       | 0.1854            |

=> As you can see they are all using float ( Norm ) and Color Scale

## Test some hyperparams to improve these 10 configs :

### 50000 Batch size ( entire dataset)

| activation | optimizer | loss                     | epochs | batch-size | learning-rate | isGray | isNorm | last_loss | last_val_loss | last_accuracy | last_val_accuracy |
|------------|-----------|--------------------------|--------|------------|---------------|--------|--------|-----------|---------------|---------------|-------------------|
| linear     | adamax    | categorical_crossentropy | 500    | 50000      | 0.0001        | FAUX   | VRAI   | 2.29883   | 2.30266       | 0.10917       | 0.0988            |
| selu       | adam      | categorical_crossentropy | 500    | 50000      | 0.0001        | FAUX   | VRAI   | 2.25644   | 2.2525        | 0.1631        | 0.1686            |
| linear     | adam      | categorical_crossentropy | 500    | 50000      | 0.0001        | FAUX   | VRAI   | 2.2529    | 2.25168       | 0.12513       | 0.1298            |
| elu        | rmsprop   | categorical_crossentropy | 500    | 50000      | 0.0001        | FAUX   | VRAI   | 2.36022   | 2.35007       | 0.09943       | 0.1025            |
| elu        | adamax    | categorical_crossentropy | 500    | 50000      | 0.0001        | FAUX   | VRAI   | 2.28577   | 2.28486       | 0.14382       | 0.1466            |
| linear     | sgd       | categorical_crossentropy | 500    | 50000      | 0.0001        | FAUX   | VRAI   | 2.27598   | 2.27837       | 0.13938       | 0.1406            |
| linear     | nadam     | categorical_crossentropy | 500    | 50000      | 0.0001        | FAUX   | VRAI   | 2.31367   | 2.29267       | 0.10797       | 0.1078            |
| softplus   | rmsprop   | categorical_crossentropy | 500    | 50000      | 0.0001        | FAUX   | VRAI   | 2.30084   | 2.30058       | 0.1144        | 0.1070            |

=> Results are not good, it seems lower batch size performs better with 500 epochs

### Change Learning Rate

| activation | optimizer | loss                     | epochs | batch-size | learning-rate | isGray | isNorm | last_loss | last_val_loss | last_accuracy | last_val_accuracy |
|------------|-----------|--------------------------|--------|------------|---------------|--------|--------|-----------|---------------|---------------|-------------------|
| linear	 | adam	     | categorical_crossentropy	| 500	 | 5000       |	0.0005	      | FAUX   | VRAI	| 2.03808	| 2.053	        | 0.19168	    |0.1879             |             
| selu	     | adam	     | categorical_crossentropy	| 500	 | 5000       |	0.0005	      | FAUX   | VRAI	| 2.06036	| 2.06925	    | 0.19103	    |0.1869             |
| elu	     | adamax	 | categorical_crossentropy	| 500	 | 5000       |	0.0005	      | FAUX   | VRAI	| 2.10677	| 2.11317	    | 0.1832	    |0.1841             |
| linear	 | adamax	 | categorical_crossentropy	| 500	 | 5000       |	0.0005	      | FAUX   | VRAI	| 2.09041	| 2.09576	    | 0.1891	    |0.1828             |
| linear	 | adam	     | categorical_crossentropy	| 500	 | 5000       |	0.001	      | FAUX   | VRAI	| 2.05202	| 2.06735	    | 0.1851	    |0.1766             |
| linear	 | sgd	     | categorical_crossentropy	| 500	 | 5000       |	0.001	      | FAUX   | VRAI	| 2.16486	| 2.16485	    | 0.1732	    |0.1765             |
| selu	     | adam	     | categorical_crossentropy	| 500	 | 5000       |	0.001	      | FAUX   | VRAI	| 2.15891	| 2.1626	    | 0.17042	    |0.1738             |
| linear	 | nadam	 | categorical_crossentropy	| 500	 | 5000       |	0.001	      | FAUX   | VRAI	| 2.07415	| 2.08653	    | 0.1716	    |0.1703             |
| linear	 | nadam	 | categorical_crossentropy	| 500	 | 5000       |	0.0005	      | FAUX   | VRAI	| 2.1545	| 2.16839	    | 0.17083	    |0.1621             |
| elu	     | rmsprop	 | categorical_crossentropy	| 500	 | 5000       |	0.001	      | FAUX   | VRAI	| 2.18739	| 2.19589	    | 0.1644	    |0.1561             |
| elu	     | rmsprop	 | categorical_crossentropy	| 500	 | 5000       |	0.0005	      | FAUX   | VRAI	| 2.21895	| 2.23468	    | 0.17537	    |0.1561             |
| elu	     | adamax	 | categorical_crossentropy	| 500	 | 5000       |	0.001	      | FAUX   | VRAI	| 2.18876	| 2.18864	    | 0.1582	    |0.1543             |
| softplus	 | rmsprop	 | categorical_crossentropy	| 500	 | 5000       |	0.0005	      | FAUX   | VRAI	| 2.26413	| 2.26003	    | 0.15552	    |0.1469             |
| linear	 | sgd	     | categorical_crossentropy	| 500	 | 5000       |	0.0005	      | FAUX   | VRAI	| 2.29088	| 2.29132	    | 0.11375	    |0.1147             |
| linear	 | adamax	 | categorical_crossentropy	| 500	 | 5000       |	0.001	      | FAUX   | VRAI	| 2.28205	| 2.28255	    | 0.1128	    |0.1145             |
| softplus	 | rmsprop	 | categorical_crossentropy	| 500	 | 5000       |	0.001	      | FAUX   | VRAI	| 2.30265	| 2.3028	    | 0.10087	    |0.0946             |
 
> changing learning rate don't gives better result


# Neural Networks ( Fully connected Dense )

As we have seen with Linear Norm is better because it avoid Nan, and Gray don't gives better results.
> For NN we will only use Color Scale and Normalized data

## Test all combinaisons of Activation / Optimizers with:
- learning rate : 0.0001
- batchs : 5000 / 1000
- epochs : 500
- Norm : True
- Gray : False
- loss : categorical_crossentropy
- layers : 32-32-32-32


#### Best 10 configs for NN

| activation | optimizer | loss	                    | epochs |batch-size  |	learning-rate |	layers	    |isGray	 |isNorm	 |   Dropout	| L1	| L2	| last_loss	| last_val_loss	| last_accuracy	| last_val_accuracy |
|------------|-----------|--------------------------|--------|------------|---------------|-------------|--------|-----------|--------------|-------|-------|-----------|---------------|---------------|-------------------|
|elu	     | adam	     | categorical_crossentropy	| 100	 | 1000	      | 0.0001	      |32-32-32-32	| FAUX	 | VRAI	     |0	            |0	    |0	    |1.23077    | 1.456	        | 0.5611    	| 0.4903            |
|softplus	 | adam	     | categorical_crossentropy	| 100	 | 1000	      | 0.0001	      |32-32-32-32	| FAUX	 | VRAI	     |0	            |0	    |0	    |1.34584    | 1.47875	    | 0.5206	    | 0.4806            |
|elu	     | adamax	 | categorical_crossentropy	| 100	 | 1000	      | 0.0001	      |32-32-32-32	| FAUX	 | VRAI	     |0	            |0	    |0	    |1.32575    | 1.48325	    | 0.53212	    | 0.4797            |
|softsign	 | adam      | categorical_crossentropy	| 100	 | 1000	      | 0.0001	      |32-32-32-32	| FAUX	 | VRAI	     |0	            |0	    |0	    |1.20283    | 1.532	        | 0.57162	    | 0.4758            |
|selu	     | adam	     | categorical_crossentropy	| 100	 | 1000	      | 0.0001	      |32-32-32-32	| FAUX	 | VRAI	     |0	            |0	    |0	    |1.3193	    | 1.51017	    | 0.53282	    | 0.4726            |
|selu	     | adamax	 | categorical_crossentropy	| 100	 | 1000	      | 0.0001	      |32-32-32-32	| FAUX	 | VRAI	     |0	            |0	    |0	    |1.32171    | 1.51393	    | 0.53057	    | 0.4726            |
|softplus	 | adam	     | categorical_crossentropy	| 100	 | 1000	      | 0.0001	      |32-32-32-32	| FAUX	 | VRAI	     |0	            |0	    |0	    |1.32127    | 1.52043	    | 0.53282	    | 0.4704            |
|elu	     | nadam	 | categorical_crossentropy	| 100	 | 1000	      | 0.0001	      |32-32-32-32	| FAUX	 | VRAI	     |0	            |0	    |0	    |1.29081    | 1.52104	    | 0.5403	    | 0.4672            |
|selu	     | nadam	 | categorical_crossentropy	| 100	 | 1000	      | 0.0001	      |32-32-32-32	| FAUX	 | VRAI	     |0	            |0	    |0	    |1.36742    | 1.52614	    | 0.51788	    | 0.4665            |
|softplus	 | damax	 | categorical_crossentropy	| 100	 | 1000	      | 0.0001	      |32-32-32-32	| FAUX	 | VRAI	     |0	            |0	    |0	    |1.40602    | 1.52741	    | 0.50178	    | 0.4645            |



> Accuracy is  better compared to Linear models.<br>
> We are close to 0.5 Accuracy.<br>
> We have only a little overfit.

## Test some hyperparams and layers structure to improve these 10 configs  :

### Layers Structure
- 64-64-64-64
- 128-128-128-128
- 128-128-128-128-128-128
- 255-255-255-255

#### Results (Best 10)
| activation | optimizer | loss	                    | epochs |batch-size  |	learning-rate |	layers	        |isGray	 |isNorm     | Dropout	| L1	| L2	| last_loss	| last_val_loss	| last_accuracy	| last_val_accuracy |
|------------|-----------|--------------------------|--------|------------|---------------|-----------------|--------|-----------|----------|-------|-------|-----------|---------------|---------------|-------------------|
| elu	     | adamax	 | categorical_crossentropy	| 500	 | 1000	      |  0.0001	      | 256-256-256-256 | FAUX	 | VRAI	     | 0        | 0	    | 0     | 0.8182	| 1.44976	    |0.71527	    | 0.5366            |
| selu	     | adam	     | categorical_crossentropy	| 500	 | 1000	      |  0.0001	      | 256-256-256-256 | FAUX	 | VRAI	     | 0        | 0	    | 0     | 0.62829	| 1.66146	    |0.7786	        | 0.5267            |
| elu	     | adam	     | categorical_crossentropy	| 500	 | 1000	      |  0.0001	      | 256-256-256-256 | FAUX	 | VRAI	     | 0        | 0	    | 0     | 0.9034	| 1.43963	    |0.68015	    | 0.5265            |
| elu	     | adamax	 | categorical_crossentropy	| 500	 | 1000	      |  0.0001	      | 128-128-128-128 | FAUX	 | VRAI	     | 0        | 0	    | 0     | 1.02173	| 1.41011	    |0.64025	    | 0.5243            |
| softplus   | adam	     | categorical_crossentropy	| 500	 | 1000	      |  0.0001	      | 256-256-256-256 | FAUX	 | VRAI	     | 0        | 0	    | 0     | 0.99166	| 1.43647	    |0.6437	        | 0.5236            |
| selu	     | adamax	 | categorical_crossentropy	| 500	 | 1000	      |  0.0001	      | 256-256-256-256 | FAUX	 | VRAI	     | 0        | 0	    | 0     | 0.87139	| 1.4488	    |0.69845	    | 0.5229            |
| elu	     | adam	     | categorical_crossentropy	| 500	 | 1000	      |  0.0001	      | 128-128-128-128 | FAUX	 | VRAI	     | 0        | 0	    | 0     | 0.95531	| 1.45568	    |0.6622   	    | 0.5219            |
| softplus   | adamax	 | categorical_crossentropy	| 500	 | 1000	      |  0.0001	      | 256-256-256-256 | FAUX	 | VRAI	     | 0        | 0	    | 0     | 1.1161	| 1.40723	    |0.6042	        | 0.5215            |
| softplus   | adam	     | categorical_crossentropy	| 500	 | 1000	      |  0.0001	      | 128-128-128-128 | FAUX	 | VRAI	     | 0        | 0	    | 0     | 1.01214	| 1.46002	    |0.63945	    | 0.5208            |
| elu	     | nadam	 | categorical_crossentropy	| 500	 | 1000	      |  0.0001	      | 128-128-128-128 | FAUX	 | VRAI	     | 0        | 0	    | 0     | 0.89541	| 1.51284	    |0.68068	    | 0.5193            |

> We can see no structure with 8 layers of 128 is in the top 10. We can suppose increasing the number of neurons as a better impact on results.
<br>
> Models are now overfitting  with almost 0.20 difference between train and test.

### Layer Structure 2 ( Let's test more neurons)
- 512-512-512-512
- 1024-1024-1024-1024

| activation | optimizer | loss	                    | epochs |batch-size  |	learning-rate |	layers	           |isGray	 |isNorm    | Dropout  | L1	   | L2	   | last_loss | last_val_loss | last_accuracy | last_val_accuracy |
|------------|-----------|--------------------------|--------|------------|---------------|--------------------|--------|-----------|----------|-------|-------|-----------|---------------|---------------|-------------------|
| selu	     | adam	     | categorical_crossentropy	| 500	 | 1000	      | 0.0001	      | 512-512-512-512	   | FAUX	| VRAI	    | 0	       | 0	   | 0	   | 0.23604   | 2.18678	   | 0.9247	       | 0.5386            |
| softplus	 | nadam	 | categorical_crossentropy	| 500	 | 1000	      | 0.0001	      | 512-512-512-512	   | FAUX	| VRAI	    | 0	       | 0	   | 0	   | 0.24998   | 2.87626	   | 0.92032	   | 0.4832            |
| softsign	 | adam	     | categorical_crossentropy	| 500	 | 1000	      | 0.0001	      | 1024-1024-1024-1024| FAUX	| VRAI	    | 0	       | 0	   | 0	   | 0.3167	   | 1.94076	   | 0.89882	   | 0.5244            |
| selu	     | adamax	 | categorical_crossentropy	| 500	 | 1000	      | 0.0001	      | 1024-1024-1024-1024| FAUX	| VRAI	    | 0	       | 0	   | 0	   | 0.38091   | 1.72022	   | 0.87552	   | 0.5458            |
| elu	     | adam	     | categorical_crossentropy	| 500	 | 1000	      | 0.0001	      | 1024-1024-1024-1024| FAUX	| VRAI	    | 0	       | 0	   | 0	   | 0.39739   | 2.00565	   | 0.86193	   | 0.5246            |

> Let's try to add regularizers to improve val_accuracy on models with higher overfit

### Add regularizers to avoid overfiting

selu	adam	categorical_crossentropy	500	1000	0.0001	512-512-512-512	FAUX	VRAI	0	0	0	0.23604	2.18678	0.9247	0.5386
softplus	nadam	categorical_crossentropy	500	1000	0.0001	512-512-512-512	FAUX	VRAI	0	0	0	0.24998	2.87626	0.92032	0.4832
softsign	adam	categorical_crossentropy	500	1000	0.0001	1024-1024-1024-1024	FAUX	VRAI	0	0	0	0.3167	1.94076	0.89882	0.5244
selu	adamax	categorical_crossentropy	500	1000	0.0001	1024-1024-1024-1024	FAUX	VRAI	0	0	0	0.38091	1.72022	0.87552	0.5458
elu	adam	categorical_crossentropy	500	1000	0.0001	1024-1024-1024-1024	FAUX	VRAI	0	0	0	0.39739	2.00565	0.86193	0.5246



