import pandas as pd
import numpy as np
from datetime import datetime


class I_125:
  """
  Class of Source I_125
  """
  def __init__(self, CalDate,Sk = 0.635):
    # cGy/h/U, cGy cm2/h
    DoseRateConstant = 0.965
    #Air kerma stength cGycm^2/h
    #conversion factor 1.270U/mCi
    #Activity = 0.5 mCi

    #Sk = 1.270U/mCi * 0.5 mCi  =0.635
    
    length=2.8 #mm

    MeanLife=59.4 # dias

    #phyani(r)
    Phyani=pd.DataFrame([
    [0.10,	1.234],
    [0.15,	1.233],
    [0.25,	1.034],
    [0.50,	0.943],
    [0.75,	0.932],
    [1.00,	0.931],
    [2.00,	0.934],
    [3.00,	0.938],
    [4.00,	0.940],
    [5.00,	0.944],
    [7.50,	0.945],
    [10.0,	0.948]],columns=['r(cm)','phi(r)']
    )

    # g(r) (gl(r) for r<rmin;gp(r) for r>=rmin)
    RadialDoseFuntion = pd.DataFrame([
    [0.05,	1.147],
    [0.06,	1.097],
    [0.07,	1.081],
    [0.08,	1.077],
    [0.09,	1.077],
    [0.10,	1.078],
    [0.15,	1.086],
    [0.20,	1.093],
    [0.25,	1.094],
    [0.30,	1.092],
    [0.40,	1.083],
    [0.50,	1.073],
    [0.60,	1.061],
    [0.70,	1.047],
    [0.75,	1.040],
    [0.80,	1.032],
    [0.90,	1.017],
    [1.00,	1.000 ],
    [1.50,	0.913	],
    [2.00,	0.819	],
    [2.50,	0.726	],
    [3.00,	0.639	],
    [3.50,	0.558	],
    [4.00,	0.486	],
    [4.50,	0.421	],
    [5.00,	0.364	],
    [5.50,	0.315	],
    [6.00,	0.271	],
    [6.50,	0.233	],
    [7.00,	0.201	],
    [7.50,	0.172	],
    [8.00,	0.148	],
    [8.50,	0.126	],
    [9.00,	0.108	],
    [9.50,	0.093	],
    [10.0,	0.079]],columns=['r(cm)','g(r)'])
    
    self.Sk=Sk
    self.DoseRateConstant=DoseRateConstant
    self.Phyani=Phyani
    self.RadialDoseFuntion=RadialDoseFuntion
    self.length=length
    self.CalDate=CalDate #
    self.MeanLife=MeanLife