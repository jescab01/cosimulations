# cosimulations
Some developments for a basic brain network simulation mixing Jansen-Rit and Wilson-Cowan models. 

model script - **JansenRit_WilsonCowan.py**
- The model combines the equations of both neural masses

coupling script - **coupling_JRWC.py**
- Adds a function "SigmoidalJansenRit_Linear" combining linear coupling for Wilson-Cowan nodes and SigmoidalJansenRit for Jansen-Rit nodes. 

Main execution script - **.JansenRit_WilsonCowan_ThalamocorticalNetwork.py**
- Script prepared to test the new model with TVB. 
