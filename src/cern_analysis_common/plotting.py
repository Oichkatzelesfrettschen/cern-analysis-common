import matplotlib.pyplot as plt
import mplhep as hep

def set_hep_style(experiment="ALICE", **kwargs):
    """
    Set matplotlib style to standard HEP experiment style.
    
    Args:
        experiment (str): "ALICE", "ATLAS", "CMS", "LHCb".
        **kwargs: Arguments passed to hep.style.use()
    """
    try:
        style = getattr(hep.style, experiment)
        plt.style.use(style)
    except AttributeError:
        plt.style.use(hep.style.ROOT)
        
def add_experiment_label(experiment="ALICE", data=True, lumi=None, year=None):
    """
    Add experiment label to the current plot.
    """
    hep.label.exp_label(exp=experiment, data=data, lumi=lumi, year=year)
