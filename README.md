# quantifying-uncertainty
Implementation of LeNet using Dirichlet distribution to enable quantifying uncertainty

To train, run the following from the command line:  
python quantify_classification_uncertainty.py --train --eqfive --epochs 20  
<br/>
The argument --eqfive can be replaced with --eqfour or --eqthree (these are the three loss functions mentioned in the paper).  
<br/>
To test, run the following:  
python quantify_classification_uncertainty.py --test --eqfive
