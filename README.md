## Documentation
- Please check the documentation of [tensorflow compression](https://tensorflow.github.io/compression/)       
- All code and notebooks are in [code/](https://github.com/BasemElbarashy/image-compression-and-semantic-segmentation/edit/master/code/)      
- Detaild documentation [code/demo.ipynb](https://github.com/BasemElbarashy/image-compression-and-semantic-segmentation/blob/master/code/demo.ipynb)                      
- colorlog used so that logs is colored and look different than tensorflow logs               
            
```bash
pip install colorlog
```
            

## To run an experiment (train different models with different lambdas)
This command will create a directory inside experiments/ to save all output files of the experiment. And for this experiement a new directory will be created for each lambda and inside this directory all model files will be saved along with pickle file that carry values of test metrics (MSE, MSSSIM, ...) and also the args used to create the experiement. You will also have directory with all images compressed and reconstructed and their metrics saved as pickle file. time_analysis.txt will contain time taken for training and for testing
```bash
python examples/run_exp.py -v --train_glob="/home/fyang/data/compression_dataset/professional/train/*.png" 
--last_step=5000 --test_glob="/home/fyang/data/compression_dataset/professional/valid/*.png" 
--exp_name exp_pro_f64 --lambdas 128,256,512,1024 --num_filters 64 --gpu 5,1 exp 
--patchsize 240 --outdir "experiments/" 2>&1 | tee logs.txt
```   

## To plot metric curves for one or more experiemnets and display results table  

You can do metric plots (MSE/MSSSIM,MSSSIM_db/PSNR vs bbp) for one or several experiements on the same plot and the will be saved in outDir/, and you can also use the command line for getting plots
 

```python
from plot_graphs import plot_graphs
# The four plots for one experiment 
plot_graphs(expsDir=['../experiments/exp_cityscapes_filter128/'], outDir='../experiments/')
# The four plots for two experiments on same figure 
plot_graphs(expsDir=['../experiments/exp_cityscapes_filter128/', '../experiments/exp_cityscapes_filter64/'], outDir='../experiments/')
```           

You will also get the results table as csv file and as png  

#### Example oupput:
<p align="center">
 <img src=figures/bppVsMSSSIM.png?raw=true>
</p>

<p align="center">
 <img src=figures/results.png?raw=true>
</p>
                 


## Notes
* mse is not scaled with 255**2 while training and that is why bigger lambdas used than ones mentioned in the main repo.     
* The training image dimension should be multiple of 16, so all training images are resized (upscaled) to the nearst resolution that achieve that  
