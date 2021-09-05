
#  Author: Shahab Zare 
#  Email: seshazahoma2@gmail.com
#  All rights reserved

# import libraries
import os

dir = os.getcwd()
from eppy import modeleditor
from eppy.modeleditor import IDF
iddfile = dir + '/Energy+.idd'
try:
    IDF.setiddname(iddfile)
except modeleditor.IDDAlreadySetError as e:
    pass


idfname = dir + 'model/model.idf'
epwfile = dir + '4format/IRN_QZ_Qazvin.AP.407310_TMYx.2003-2017EPW.epw'

# create model
idf = IDF(idfname, epwfile)

# placeholder for zone
z = idf.idfobjects['Zone']
zones = [z[i].Name for i in range(len(z))]
areas = [z[i].Floor_Area for i in range(len(z))]
AreaDict = dict(zip(zones,areas))

# placeholder for material
m = idf.idfobjects['Material']

# placeholder for glazing
wmg = idf.idfobjects[ 'WindowMaterial:Glazing' ]

# placeholder for gas
wmgs = idf.idfobjects[ 'WindowMaterial:Gas' ]

# placeholder for no mass
nm = idf.idfobjects[ 'Material:NoMass' ]

import functools

def call_once(func):
    @functools.wraps(func)
    def wrapper_call_once(*args, **kwargs):
        wrapper_call_once.num_calls += 1
        if wrapper_call_once.num_calls == 1:
            return func(*args, **kwargs)
        elif wrapper_call_once.num_calls > 1:
            pass

    wrapper_call_once.num_calls = 0
    
    return wrapper_call_once

# add materials
@call_once
def AddDoorMaterial(idf,m,nm):
    idf.newidfobject('Material')
    m[-1].Name = '19_1_36'
    m[-1].Roughness = 'Rough'
    m[-1].Thickness = 0.006
    m[-1].Conductivity = 0.15
    m[-1].Density = 700
    m[-1].Specific_Heat = 1420
    m[-1].Thermal_Absorptance =  0.9 
    m[-1].Solar_Absorptance =  0.78 
    m[-1].Visible_Absorptance =  0.78 
    
    # No Mass

    nm = idf.idfobjects[ 'Material:NoMass' ]
    idf.newidfobject( 'Material:NoMass' )
    nm[-1].Name =  '19_RVAL_2' 
    nm[-1].Thermal_Resistance =  0.15 
    nm[-1].Thermal_Absorptance =  0.9 
    nm[-1].Solar_Absorptance =  0.7 
    nm[-1].Visible_Absorptance =  0.7 
    
    idf.newidfobject( 'Material:NoMass' )
    nm[-1].Name =  '20_RVAL_2' 
    nm[-1].Thermal_Resistance =  0.15 
    nm[-1].Thermal_Absorptance =  0.9 
    nm[-1].Solar_Absorptance =  0.7 
    nm[-1].Visible_Absorptance =  0.7 
    return 1

@call_once
def AddXPS(idf,m): 
    idf.newidfobject( 'Material' )
    m[-1].Name =  'XPS' 
    m[-1].Roughness =  'Rough' 
    m[-1].Thickness = 0.02
    m[-1].Conductivity =  0.034 
    m[-1].Density =  35 
    m[-1].Specific_Heat =  1400 
    m[-1].Thermal_Absorptance =  0.9 
    m[-1].Solar_Absorptance =  0.6 
    m[-1].Visible_Absorptance =  0.6 
    return 1

@call_once
def AddRockwoolT(idf,m):
    idf.newidfobject( 'Material' )
    m[-1].Name =  'Rock-wool-100'
    m[-1].Roughness =  'Rough' 
    m[-1].Thickness =  0.02 
    m[-1].Conductivity =  0.033 
    m[-1].Density =  100 
    m[-1].Specific_Heat =  710 
    m[-1].Thermal_Absorptance =  0.9 
    m[-1].Solar_Absorptance =  0.6 
    m[-1].Visible_Absorptance =  0.6
    return 1

@call_once
def AddRockwoolE(idf,m):
    idf.newidfobject( 'Material' )
    m[-1].Name =  'Rock-wool-80' 
    m[-1].Roughness =  'Rough' 
    m[-1].Thickness =  0.02 
    m[-1].Conductivity =  0.033 
    m[-1].Density =  80 
    m[-1].Specific_Heat =  710 
    m[-1].Thermal_Absorptance =  0.9 
    m[-1].Solar_Absorptance =  0.6 
    m[-1].Visible_Absorptance =  0.6 
    return 1

@call_once
def AddPVC(idf,m):
    idf.newidfobject( 'Material' )
    m[-1].Name =  'PVC'
    m[-1].Roughness =  'Rough' 
    m[-1].Thickness =  0.08 
    m[-1].Conductivity =  0.016 
    m[-1].Density =  1379 
    m[-1].Specific_Heat =  1004 
    m[-1].Thermal_Absorptance =  0.9 
    m[-1].Solar_Absorptance =  0.6 
    m[-1].Visible_Absorptance =  0.6 
    return 1

@call_once
def AddEPST(idf,m):
    idf.newidfobject( 'Material' )
    m[-1].Name =  'EPS-20' 
    m[-1].Roughness =  'Rough' 
    m[-1].Thickness =  0.02 
    m[-1].Conductivity =  0.035 
    m[-1].Density =  20 
    m[-1].Specific_Heat =  1400 
    m[-1].Thermal_Absorptance =  0.9 
    m[-1].Solar_Absorptance =  0.6 
    m[-1].Visible_Absorptance =  0.6 
    return 1

@call_once
def AddEPSE(idf,m):
    idf.newidfobject( 'Material' )
    m[-1].Name =  'EPS-8' 
    m[-1].Roughness =  'Rough' 
    m[-1].Thickness =  0.02 
    m[-1].Conductivity =  0.035 
    m[-1].Density =  8 
    m[-1].Specific_Heat =  1400 
    m[-1].Thermal_Absorptance =  0.9 
    m[-1].Solar_Absorptance =  0.6 
    m[-1].Visible_Absorptance =  0.6 
    return 1

@call_once
def AddCement(idf,m):
    # Cement
    idf.newidfobject( 'Material' )
    m[-1].Name =  '3_1_5135' 
    m[-1].Roughness =  'Rough' 
    m[-1].Thickness =  0.01 
    m[-1].Conductivity =  0.72 
    m[-1].Density =  1860 
    m[-1].Specific_Heat =  840 
    m[-1].Thermal_Absorptance =  0.9 
    m[-1].Solar_Absorptance =  0.6 
    m[-1].Visible_Absorptance =  0.6
    return 1

@call_once
def AddPlaster(idf,m):
    # Gypsum Plaster
    idf.newidfobject('Material' )
    m[-1].Name =  '1_6_67' 
    m[-1].Roughness =  'Rough' 
    m[-1].Thickness =  0.005 
    m[-1].Conductivity =  0.4 
    m[-1].Density =  1000 
    m[-1].Specific_Heat =  1000 
    m[-1].Thermal_Absorptance =  0.9 
    m[-1].Solar_Absorptance =  0.5 
    m[-1].Visible_Absorptance =  0.5 
    return 1

@call_once
def AddWinMaterial(idf,wmg,wmgs):
    # Window Glazing
    idf.copyidfobject(wmg[0])
    wmg[-1].Name =  '63'  
    wmg[-1].Thickness =  0.004 
    wmg[-1].Solar_Transmittance_at_Normal_Incidence =  0.816 
    wmg[-1].Visible_Transmittance_at_Normal_Incidence =  0.892 
    wmg[-1].Conductivity =  1 
    
    idf.newidfobject( 'WindowMaterial:Gas' )
    wmgs[-1].Name =  '1017' 
    wmgs[-1].Thickness =  0.01 
    wmgs[-1].Gas_Type =  'Air'
    
    idf.newidfobject( 'WindowMaterial:Gas' )
    wmgs[-1].Name =  'Half thickness 1017' 
    wmgs[-1].Thickness =  0.005 
    wmgs[-1].Gas_Type =  'Air'
    
    idf.newidfobject( 'WindowMaterial:Gas' )
    wmgs[-1].Name =  '1016' 
    wmgs[-1].Thickness =  0.01 
    wmgs[-1].Gas_Type =  'Argon' 
    
    idf.newidfobject( 'WindowMaterial:Gas' )
    wmgs[-1].Name =  'Half thickness 1016'
    wmgs[-1].Thickness =  0.005 
    wmgs[-1].Gas_Type =  'Argon' 
    return 1
    
    
    
import numpy as np

# decode input vector
def decoder(InArray, show_results = False):
    assert len(InArray)==22 ,'Incorrect input dimension! it must be 22'
    for i in range(len(InArray)):
        assert InArray[i] >= 0 and InArray[i] <= 1,'Input arrays should be normalized'
    
    # InArray = [A1 ,... , A38]
    
    offset = 10**(-8)
    
    # A1: Cool Set Point
    perms = 4  #Permutations
    mod = int(perms*float(InArray[0]) - offset) 
    Cool = mod + 24  #[24,25,26,27]
    
    
    # A2: Heat Set Point
    mod =int(perms*float(InArray[1]) - offset)
    Heat = mod + 19 #[19,20,21,22]
        
    
    # A[3:6]: Floor
    Thickness = np.linspace(0.0001,0.1,11).tolist()
    Ins = range(1,4)
    
    perms = len(Thickness)*len(Ins)
    mod =[int(perms*float(InArray[i]) - offset) for i in range(2,6)]
    floorinsID = []
    floorthickness = []
    for item in mod:
        q,r = divmod(item,len(Thickness))
        floorinsID.append(Ins[q])
        floorthickness.append(Thickness[r])
    
    # A7: Roof 
    # We have one roof
    
    perms = len(Thickness)*len(Ins)
    mod =int(perms*float(InArray[6]) - offset)
    
    q,r = divmod(mod,len(Thickness))
    roofinsID = Ins[q]
    roofthickness = Thickness[r]
    
    
    # A[8:12]: Wall 
    Ins = range(1,6)
    cType = range(1,4)
    perms = len(Thickness)*len(Ins)*len(cType)
    mod =[int(perms*float(InArray[i]) - offset) for i in range(7,12)]
    wallinsID = []
    wallthickness = []
    wallcType = []
    for item in mod:
        q,r = divmod(item,len(Thickness))
        wallthickness.append(Thickness[r])
        q,r = divmod(q,len(cType))
        wallcType.append(cType[r])
        q,r = divmod(q,len(Ins))
        wallinsID.append(Ins[r])
    wallcType = wallcType[:2]
    
    # A[13:20]: Window
    Type = range(1,6)
    
    perms = len(Type)
    mod =[int(perms*float(InArray[i]) - offset) for i in range(12,20)]
    windowType = []
    for item in mod:
        windowType.append(Type[item])
    
    # A[21:22]: Door
    mod = [float(InArray[20]) , float(InArray[21])]
    doorConst = []
    for i in range(2):
        if mod[i] < 0.5:
            doorConst.append(1)
        else:
            doorConst.append(2)
        
    Data = [Cool, Heat, floorinsID, floorthickness, roofinsID, roofthickness, wallinsID, wallcType, wallthickness, windowType, doorConst]

    if show_results:
        import inspect

        def retrieve_name(var):
            callers_local_vars = inspect.currentframe().f_back.f_locals.items()
            return [var_name for var_name, var_val in callers_local_vars if var_val is var]
        
        
        for i in range(len(Data)):
            print(f'Dimension {i + 1} , {retrieve_name(Data[i])[0]} is {Data[i]}  \n')
        
    return Data
    
# manipulate idf file
def FloorActuator(floorID,thickness,AreaDict):
    '''
    ID     Zone           Construction     Floor
    -----------------------------------------------------
    1      4767,5668       13,14           Stairs 2 (4767_Floor_8_0_0)
    2      5668,6114       13,14           Stairs 1 (5668_Floor_0_0_0)
    --------------------------------------------------------
    3      4779,5661       23,24           Bathroom 2 (4786_Floor_4_0_0)
    4      4786,5654       23,24           Toilet 2  (4779_Floor_5_0_0)
    ---------------------------------------------------------
    5      4793,5642       35,36           Livingroom 2 (4793_Floor_10_0_0)(4793_Floor_10_0_1)(4793_Floor_10_0_2)
    ---------------------------------------------------------
    6      4805,5634       39,40           Kitchen 2 (4805_Floor_3_0_0)
    ---------------------------------------------------------
    7      4813,5622       41,42           BedroomZahra 2  (4813_Floor_5_0_0)
    8      4820,5618       41,42           BedroomMona 2   (4820_Floor_4_0_0)
    --------------------------------------------------------
    9      5618,6122       43,44           BedroomMona 1   (5618_Floor_0_0_0)
    10     5627,6122       43,44           BedroomZahra 1  (5627_Floor_0_0_0)
        --------------------------------------------------------
    11     5634,6122       43,44           Kitchen 1    (5634_Floor_0_0_0)
    12     5642,6122       43,44           Livingroom 1  (5642_Floor_0_0_0)(5642_Floor_0_0_1)(5642_Floor_0_0_2)
    13     5564,6122       47              Bathroom 1  (5654_Floor_0_0_0)
    14     5661,6122       47              Toilet 1   (5661_Floor_0_0_0)
    --------------------------------------------------------
    15     6114            47              Pilot     (6114_GroundFloor_0_0_0)(6114_Partition_2_0_0)
    16     6122            47              Pilot     (6122_GroundFloor_0_0_1)(6122_GroundFloor_0_0_2)
    '''
    
    floorName = ['4793_Floor_10_0_0','4793_Floor_10_0_1','4793_Floor_10_0_2','4805_Floor_3_0_0','4813_Floor_5_0_0','4820_Floor_4_0_0','5618_Floor_0_0_0','5627_Floor_0_0_0','5634_Floor_0_0_0','5642_Floor_0_0_0','5642_Floor_0_0_1','5642_Floor_0_0_2']
    
    m = idf.idfobjects['Material']
    for i in range(len(m)):
        if m[i].Name == 'XPS' or 'Rock-wool-100' or 'Rock-wool80':
            m[i].Thickness = thickness
    
                
       
    
    floor = idf.idfobjects['BuildingSurface:Detailed']
    Z =[]
    C = []
    for i in range(len(floor)):
        for j in range(len(floorName)):
            if floor[i].Name == floorName[j]:
                Z.append(floor[i].Zone_Name)
                C.append(floor[i].Construction_Name)
        
        
    c = list(set(C))  # ['39', '43', '35', '41']
    z = list(set(Z))
    
    z = [[z[4]],[z[2],z[-1],z[-3],z[0]],[z[1]],[z[3],z[-2]]]
    
    Cost = 0
    for i,j in zip(range(len(floorID)),range(len(z))):
        if len(z[j]) != 1:
            for k in range(len(z[j])):
                A = AreaDict[z[j][k]]
                if floorID[i] == 1:
                    Cost += (602480*thickness + 12466)*A*1.1 + 72364*A
                elif floorID[i] == 2:
                    Cost += (680000*thickness+ 15192)*A*1.1 + 72364*A
                else:
                    Cost += (1300000*thickness+13200)*A*1.1 + 72364*A
    return int(Cost)
    
    Cons = idf.idfobjects['construction']
    for n,m in zip(range(len(floorID)),c):
        switcher = {(floorID[n] == 1):'Rock-wool-80', (floorID[n] == 2):'Rock-wool-100', (floorID[n] == 3):'XPS'}
        for Name in range(len(Cons)):
            if n == 0:
                if Cons[Name].Name == m:
                    Cons[Name].Outside_Layer = '1_6_66'
                    Cons[Name].Layer_2 = '39_RVAL_2'
                    Cons[Name].Layer_3 = '1_4_5128'
                    Cons[Name].Layer_4 = '39_RVAL_4'
                    Cons[Name].Layer_5 = '21_5_5134' 
                    Cons[Name].Layer_6 = '1_1_10003'
                    Cons[Name].Layer_7 = '3_1_5134'
                    Cons[Name].Layer_8 = switcher[True]
                    Cons[Name].Layer_9 = '3_1_5135'
                    Cons[Name].Layer_10 = '11_1_5169'
                if Cons[Name].Name == str(int(m) + 1):
                    Cons[Name].Outside_Layer = '11_1_5169'
                    Cons[Name].Layer_2 = '3_1_5135'
                    Cons[Name].Layer_3 = switcher[True]
                    Cons[Name].Layer_4 = '3_1_5134'
                    Cons[Name].Layer_5 = '1_1_10003'
                    Cons[Name].Layer_6 = '21_5_5134' 
                    Cons[Name].Layer_7 = '39_RVAL_4'
                    Cons[Name].Layer_8 = '1_4_5128'
                    Cons[Name].Layer_9 = '39_RVAL_2'
                    Cons[Name].Layer_10 = '1_6_66'
            if n == 1:
                if Cons[Name].Name == m:
                    Cons[Name].Outside_Layer = '17_1_5'
                    Cons[Name].Layer_2 = '17_2_103'
                    Cons[Name].Layer_3 = '17_3_11'
                    Cons[Name].Layer_4 = '17_4_66'
                    Cons[Name].Layer_5 = switcher[True]
                    Cons[Name].Layer_6 = '3_1_5135'
                    Cons[Name].Layer_7 = '11_1_5169'
                if Cons[Name].Name == str(int(m) + 1):
                    Cons[Name].Outside_Layer = '11_1_5169'
                    Cons[Name].Layer_2 = '3_1_5135'
                    Cons[Name].Layer_3 = switcher[True]
                    Cons[Name].Layer_4 = '17_4_66'
                    Cons[Name].Layer_5 = '17_3_11'
                    Cons[Name].Layer_6 = '17_2_103'
                    Cons[Name].Layer_7 = '17_1_5'
            if n == 2:
                if Cons[Name].Name == m:
                    Cons[Name].Outside_Layer = '1_6_66'
                    Cons[Name].Layer_2 = '35_RVAL_2'
                    Cons[Name].Layer_3 = '1_4_5128'
                    Cons[Name].Layer_4 = '35_RVAL_4'
                    Cons[Name].Layer_5 = '3_1_5134' 
                    Cons[Name].Layer_6 = switcher[True]
                    Cons[Name].Layer_7 = '3_1_5135'
                    Cons[Name].Layer_8 = '11_1_5169'
                    Cons[Name].Layer_9 = '35_7_88'
                if Cons[Name].Name == str(int(m) + 1):
                    Cons[Name].Outside_Layer = '35_7_88'
                    Cons[Name].Layer_2 = '11_1_5169'
                    Cons[Name].Layer_3 = '3_1_5135'
                    Cons[Name].Layer_4 = switcher[True]
                    Cons[Name].Layer_5 = '3_1_5134' 
                    Cons[Name].Layer_6 = '35_RVAL_4'
                    Cons[Name].Layer_7 = '1_4_5128'
                    Cons[Name].Layer_8 = '35_RVAL_2'
                    Cons[Name].Layer_9 = '1_6_66'
            if n == 3:
                if Cons[Name].Name == m:
                    Cons[Name].Outside_Layer = '1_6_66'
                    Cons[Name].Layer_2 = '41_RVAL_2'
                    Cons[Name].Layer_3 = '1_4_5128'
                    Cons[Name].Layer_4 = '41_RVAL_4'
                    Cons[Name].Layer_5 = '3_1_5134' 
                    Cons[Name].Layer_6 = switcher[True]
                    Cons[Name].Layer_7 = '3_1_5135'
                    Cons[Name].Layer_8 = '11_1_5169'
                    Cons[Name].Layer_9 = '41_6_197'
                if Cons[Name].Name == str(int(m) + 1):
                    Cons[Name].Outside_Layer = '41_6_197'
                    Cons[Name].Layer_2 = '11_1_5169'
                    Cons[Name].Layer_3 = '3_1_5135'
                    Cons[Name].Layer_4 = switcher[True]
                    Cons[Name].Layer_5 = '3_1_5134' 
                    Cons[Name].Layer_6 = '41_RVAL_4'
                    Cons[Name].Layer_7 = '1_4_5128'
                    Cons[Name].Layer_8 = '41_RVAL_2'
                    Cons[Name].Layer_9 = '1_6_66'
                    
    

def WallActuator(insID,cType, thickness,AreaDict):
    '''
    ID    Zone  Construction   PVC    Wall  
    ----------------------------------------------------------
    1     4793       3         Yes    Livingroom-Neigbour 2    (4793_Wall_3_0_0)
    2     5642       3         Yes    Livingroom-Neigbour 1    (5642_Wall_2_0_0)
    3     4813       3         Yes    BedroomZahra-Neigbour 2  (4813_Wall_4_0_0)
    4     5627       3         Yes    BedroomZahra-Neigbour 1  (5627_Wall_2_0_0)
    5     4820       3         Yes    BedroomMona-Neigbour 2   (4820_Wall_0_0_0)
    6     5618       3         Yes    BedroomMona-Neigbour 1   (5618_Wall_6_0_0)
    ----------------------------------------------------------
    7     4793       25        Yes    Livingroom-Street 2      (4793_Wall_1_0_0)
    8     5642       25        Yes    Livingroom-Street 1      (5642_Wall_3_0_0)
    9     4813       25        Yes    BedroomZahra-Courtyard 2 (4813_Wall_4_0_0)
    10    5627       25        Yes    BedroomZahra-Courtyard 1 (5627_Wall_5_0_0)
    11    4820       25        Yes    BedroomMona-Courtyard 2  (4820_Wall_0_0_0)
    12    5618       25        Yes    BedroomMona-Courtyard 1  (5618_Wall_7_0_0)
    ------------------------------------------------------------
    13    4767       11        No     Kitchen-Stair 2          (4767_Partition_7_0_0)
    14    5634       11        No     Kitchen-Stair 1          (5634_Partition_5_0_0)
    -------------------------------------------------------------
    15    4805       23        No     Kitchen-Neigbour 2       (4805_Wall_0_0_0)
    16    5634       23        No     Kitchen-Neigbour 1       (5634_Wall_4_0_0)
    ----------------------------------------------------------
    17    4805       37        No     Kitchen-Street 2         (4805_Wall_2_0_0)
    18    5634       37        No     Kitchen-Street 1         (5634_Wall_3_0_0)

    
    '''
    WallName = ['4793_Wall_3_0_0', '5642_Wall_2_0_0', '4813_Wall_4_0_0', '5627_Wall_2_0_0', '4820_Wall_0_0_0', '5618_Wall_6_0_0', '4793_Wall_1_0_0','5642_Wall_3_0_0', '4813_Wall_4_0_0','5627_Wall_5_0_0', '4820_Wall_0_0_0', '5618_Wall_7_0_0'   '4767_Partition_7_0_0', '5634_Partition_5_0_0', '4805_Wall_0_0_0', '5634_Wall_4_0_0', '4805_Wall_2_0_0', '5634_Wall_3_0_0']
    
    
    m = idf.idfobjects['Material']
    for i in range(len(m)):
        if m[i].Name == 'XPS' or 'Rock-wool-100' or 'Rock-wool80' or 'EPS-20' or 'EPS-8':
            m[i].Thickness = thickness
    
    
    
    wall = idf.idfobjects['BuildingSurface:Detailed']
    Z =[]
    C = []
    for i in range(len(wall)):
        for j in range(len(WallName)):
            if wall[i].Name == WallName[j]:
                Z.append(wall[i].Zone_Name)
                C.append(wall[i].Construction_Name)
   
    C = list(set(C))  # ['23', '3', '11', '37', '25']
    z = list(set(Z))        
    Cons = idf.idfobjects['construction']
    
    ncwall = insID[1:3]
    cwall = [insID[0]] + insID[3:]
    
    nc = [C[-1],C[1]]  # 25 , 3
    c = [C[-2],C[0],C[2]]    # 37 , 23 , 11
    
    nz = [[z[2],z[0],z[-2],z[-1],z[4],z[3]],[z[2],z[0],z[-2],z[-1],z[4],z[3]]]
    z = [[z[1],z[-3]],[z[-4],z[-3]],[z[-4],z[-3]]]

    Cost = 0
    thickness = 0.02
    for i,j in zip(range(len(cwall)),range(len(z))):
        for k in range(len(z[j])):
            A = AreaDict[z[j][k]]
            if cwall[i] == 1:
                Cost += (602480*thickness +12466)*A*1.1 + 72364*A + 25000*A + 80000*A
            elif cwall[i] == 2:
                Cost += (680000*thickness+ 15192)*A*1.1 + 72364*A + 25000*A + 80000*A
            elif cwall[i] == 3: 
                Cost += (1300000*thickness+13200)*A*1.1 + 72364*A + 25000*A + 80000*A
            elif cwall[i] == 4: 
                Cost += (1100000*thickness+13200)*A*1.1 + 72364*A + 25000*A + 80000*A
            elif cwall[i] == 5: 
                Cost += (300000*thickness+13200)*A*1.1 + 72364*A + 25000*A + 80000*A
    for i,j in zip(range(len(ncwall)),range(len(nz))):
        for k in range(len(nz[j])):
            A = AreaDict[nz[j][k]]
            if ncwall[i] == 1:
                Cost += (602480*thickness +12466)*A*1.1 + 72364*A + 25000*A
            elif ncwall[i] == 2:
                Cost += (680000*thickness+ 15192)*A*1.1 + 72364*A + 25000*A
            elif ncwall[i] == 3: 
                Cost += (1300000*thickness+13200)*A*1.1 + 72364*A + 25000*A
            elif ncwall[i] == 4: 
                Cost += (1100000*thickness+13200)*A*1.1 + 72364*A + 25000*A 
            elif ncwall[i] == 5: 
                Cost += (300000*thickness+13200)*A*1.1 + 72364*A + 25000*A 
    return int(Cost)

    for n,m in zip(range(len(cwall)),c):
        switcher = {(cwall[n] == 1):'Rock-wool-80', (cwall[n] == 2):'Rock-wool-100', (cwall[n] == 3):'XPS', (cwall[n] == 4):'EPS-20', (cwall[n] == 5):'EPS-8'}
        for Name in range(len(Cons)):
            if n == 0:
                if Cons[Name].Name == m:
                    Cons[Name].Outside_Layer = '25_1_56'
                    Cons[Name].Layer_2 = '3_1_5134'
                    Cons[Name].Layer_3 = '3_2_5128'
                    Cons[Name].Layer_4 = '3_1_5134'
                    Cons[Name].Layer_5 = switcher[True]  
                    Cons[Name].Layer_6 = '3_1_5135'
                    Cons[Name].Layer_7 = '11_1_5169'
                if Cons[Name].Name == str(int(m) + 1):
                    Cons[Name].Outside_Layer = '11_1_5169'
                    Cons[Name].Layer_2 = '3_1_5135'
                    Cons[Name].Layer_3 = switcher[True]
                    Cons[Name].Layer_4 = '3_1_5134'
                    Cons[Name].Layer_5 = '3_2_5128'
                    Cons[Name].Layer_6 = '3_1_5134'
                    Cons[Name].Layer_7 = '25_1_56'
            if n == 1:
                if Cons[Name].Name == m:
                    Cons[Name].Outside_Layer = '3_1_5134'
                    Cons[Name].Layer_2 = '3_2_5128'
                    Cons[Name].Layer_3 = '3_1_5134'
                    Cons[Name].Layer_4 = switcher[True]
                    Cons[Name].Layer_5 = '3_1_5135'
                    Cons[Name].Layer_6 = '11_1_5169'
                if Cons[Name].Name == str(int(m) + 1):
                    Cons[Name].Outside_Layer = '11_1_5169'
                    Cons[Name].Layer_2 = '3_1_5135'
                    Cons[Name].Layer_3 = switcher[True]
                    Cons[Name].Layer_4 = '3_1_5134'
                    Cons[Name].Layer_5 = '3_2_5128'
                    Cons[Name].Layer_6 = '3_1_5134'
            if n == 2:
                if Cons[Name].Name == m:
                    Cons[Name].Outside_Layer = '11_1_5169'
                    Cons[Name].Layer_2 = '3_1_5135'
                    Cons[Name].Layer_3 = switcher[True]
                    Cons[Name].Layer_4 = '3_1_5134'
                    Cons[Name].Layer_5 = '3_2_5128'
                    Cons[Name].Layer_6 = '11_RVAL_4'
                if Cons[Name].Name == str(int(m) + 1):
                    Cons[Name].Outside_Layer = '11_RVAL_4'
                    Cons[Name].Layer_2 = '3_2_5128'
                    Cons[Name].Layer_3 = '3_1_5134'
                    Cons[Name].Layer_4 = switcher[True]
                    Cons[Name].Layer_5 = '3_1_5135'
                    Cons[Name].Layer_6 = '11_1_5169'

            
                    
    for i in range(len(ncwall)):
        switcher = {(ncwall[i] == 1):'Rock-wool-80', (ncwall[i] == 2):'Rock-wool-100', (ncwall[i] == 3):'XPS', (ncwall[i] == 4):'EPS-20', (ncwall[i] == 5):'EPS-8'}
        for Name in range(len(Cons)):
            if Cons[Name].Name == nc[i]:
                Cons[Name].Outside_Layer = '3_1_5134'
                Cons[Name].Layer_2 = '3_2_5128'
                Cons[Name].Layer_3 = '3_RVAL_3'
                Cons[Name].Layer_4 = '1_6_66'
                if cType[i] == 1:
                    Cons[Name].Layer_5 = 'PVC'
                elif cType[i] == 2:
                    Cons[Name].Layer_5 = switcher[True]
                    Cons[Name].Layer_6 = 'PVC'
                elif cType[i] == 3:
                    Cons[Name].Layer_5 = switcher[True]
                    Cons[Name].Layer_6 = '1_6_67'
            if Cons[Name].Name == str(int(nc[i]) + 1):
                if cType[i] == 1:
                    Cons[Name].Outside_Layer = 'PVC'
                    Cons[Name].Layer_2 = '1_6_66'
                    Cons[Name].Layer_3 = '3_RVAL_3'
                    Cons[Name].Layer_4 = '3_2_5128'
                    Cons[Name].Layer_5 = '3_1_5134'
                elif cType[i] == 2:
                    Cons[Name].Outside_Layer = 'PVC'
                    Cons[Name].Layer_2 = switcher[True]
                elif cType[i] == 3:
                    Cons[Name].Outside_Layer = '1_6_67'
                    Cons[Name].Layer_2 = switcher[True]
                if cType[i] != 1:
                    Cons[Name].Layer_3 = '1_6_66'
                    Cons[Name].Layer_4 = '3_RVAL_3'
                    Cons[Name].Layer_5 = '3_2_5128'
                    Cons[Name].Layer_6 = '3_1_5134'
    
    
def RoofActuator(roofins,thickness,AreaDict):
    '''
    ID     Zone      Roof
    1      4779      Toilet
    2      4786      Bathroom
    3      4793      Livingroom
    4      4805      Kitchen
    5      4813      BedroomZahra
    6      4820      BedroomMona
    '''

    m = idf.idfobjects['Material']
    for i in range(len(m)):
        if m[i].Name == 'XPS' or 'Rock-wool-100' or 'Rock-wool80':
            m[i].Thickness = thickness
    
    
    zone= [4779,4786,4793,4805,4813,4820]
    
    A = 0
    for i in range(len(zone)):
        A += AreaDict[str(zone[i])]
    
    Cost = 0
    if roofins == 1:
        Cost += (602480*thickness + 12466)*A*1.1 + 67365*A
    elif roofins == 2:
        Cost += (680000*thickness+ 15192)*A*1.1 + 67365*A
    else:
        Cost += (1300000*thickness+13200)*A*1.1 + 67365*A
    return int(Cost)

    bs = idf.idfobjects['BuildingSurface:Detailed']

    roof = [bs[i] for i in range(len(bs)) if 'Roof' in bs[i].Name]
    
    cons = idf.idfobjects['construction']
    
    for i in range(len(zone)):
        for j in range(len(roof)):
            if str(zone[i]) in roof[j].Name:
                construct = roof[j].Construction_Name
                for k in range(len(cons)):
                    if cons[k].Name == construct:
                        cons[k].Outside_Layer = '11_1_5169'
                        cons[k].Layer_2 = '3_1_5135'
                        cons[k].Layer_4 = '3_1_5134'
                        cons[k].Layer_5 = '1_1_10003'
                        cons[k].Layer_6 = '1_2_5134'
                        cons[k].Layer_7 = '1_RVAL_3'
                        cons[k].Layer_8 = '1_4_5128'
                        cons[k].Layer_9 = '1_RVAL_5'
                        cons[k].Layer_10 = '1_6_66'
                        if roofins == 1:
                            cons[k].Layer_3 = 'XPS'
                        elif roofins == 2:
                            cons[k].Layer_3 ='Rock-wool-80'
                        elif roofins == 3:
                            cons[k].Layer_3 = 'Rock-wool-100'
                    if cons[k].Name == str(int(construct)+1):
                        cons[k].Layer_10 = '11_1_5169'
                        cons[k].Layer_9 = '3_1_5135'
                        cons[k].Layer_7 = '3_1_5134'
                        cons[k].Layer_6 = '1_1_10003'
                        cons[k].Layer_5 = '1_2_5134'
                        cons[k].Layer_4 = '1_RVAL_3'
                        cons[k].Layer_3 = '1_4_5128'
                        cons[k].Layer_2 = '1_RVAL_5'
                        cons[k].Outside_Layer == '1_6_66'
                        if roofins == 1:
                            cons[k].Layer_8 = 'XPS'
                        elif roofins == 2:
                            cons[k].Layer_8 = 'Rock-wool-80'
                        elif roofins == 3:
                            cons[k].Layer_8 = 'Rock-wool-100'

def WindowActuator(Type):
    '''
    ID      Zone      Room
    1       5642      Floor 1 LivingRoom
    2       4793      Floor 2 LivingRoom
    3       5634      Floor 1 Kitchen
    4       4805      Floor 2 Kitchen
    5       5627      Floor 1 BedroomZahra
    6       4813      Floor 2 BedroomZahra
    7       5618      Floor 1 BedroomMona
    8       4820      Floor 2 BedroomMona
    '''
    
    zone = [5642,4793,5634,4805,5627,4813,5618,4820]
    win = idf.idfobjects['FenestrationSurface:Detailed']
    
    Cost = 0
    for i in range(len(zone)):
        if Type == 2:
            Cost += 123000
        elif Type == 3:
            Cost += 1100000
        elif Type == 4:
            Cost += 1767000
        elif Type == 5:
            Cost += 1557000
    return Cost
    
    
    w = [win[i] for i in range(len(win)) if 'Win' in win[i].Name and 'Wall' in win[i].Name]
    Cons = idf.idfobjects['construction']
    
    Zinf = idf.idfobjects['ZoneInfiltration:DesignFlowRate']
    DFR = [0.00502, 0.00502, 0.001975, 0.001975, 0.0015, 0.0015, 0.0015, 0.0015]  #Design Flow Rate for each zone
    for i in range(len(Type)):
        window = [w[j] for j in range(len(w)) if str(zone[i]) in w[j].Name]
        cons = window[0].Construction_Name
        conss = str(int(cons) + 1000)
        for k in range(len(Cons)):
            if Cons[k].Name == cons:
                if Type[i] == 1:
                    pass
                elif Type[i] == 2:
                    Cons[k].Outside_Layer = '63'
                    Cons[k].Layer_2 = '1017'
                    Cons[k].Layer_3 = '3'
                elif Type[i] == 3:
                    Cons[k].Outside_Layer = '63'
                    Cons[k].Layer_2 = '1017'
                    Cons[k].Layer_3 = '63'
                elif Type[i] == 4:
                    Cons[k].Outside_Layer = '63'
                    Cons[k].Layer_2 = '1016'
                    Cons[k].Layer_3 = '3'
                elif Type[i] == 5:
                    Cons[k].Outside_Layer = '63'
                    Cons[k].Layer_2 = '1016'
                    Cons[k].Layer_3 = '63'
            if Cons[k].Name == conss:
                if Type[i] == 1:
                    pass
                elif Type[i] == 2:
                    Cons[k].Outside_Layer = '63'
                    Cons[k].Layer_2 = '1017'
                    Cons[k].Layer_3 = '3'
                    Cons[k].Layer_4 = '20031'
                elif Type[i] == 3:
                    Cons[k].Outside_Layer = '63'
                    Cons[k].Layer_2 = '1017'
                    Cons[k].Layer_3 = '63'
                    Cons[k].Layer_4 = '20031'
                elif Type[i] == 4:
                    Cons[k].Outside_Layer = '63'
                    Cons[k].Layer_2 = '1016'
                    Cons[k].Layer_3 = '3'
                    Cons[k].Layer_4 = '20031'
                elif Type[i] == 5:
                    Cons[k].Outside_Layer = '63'
                    Cons[k].Layer_2 = '1016'
                    Cons[k].Layer_3 = '63'
                    Cons[k].Layer_4 = '20031'
        if Type[i] != 1:
            for n in range(len(Zinf)):
                if str(zone[i]) in Zinf[n].Name:
                    Zinf[n].Design_Flow_Rate = DFR[i]
        

from eppy.results import readhtml # the eppy module with functions to read the html

# Define EnergyPlus Objective Function
def EplusObjf(Input):
    
    D = decoder(Input)
    
    Cool = D[0]  
    Heat = D[1]  
    floorinsID = D[2]
    floorthickness = D[3][0]
    roofinsID = D[4]
    roofthickness = D[5]
    wallinsID = D[6]
    wallcType = D[7]
    wallthickness = D[8][0]
    windowType = D[9]
    doorConst = D[10]
    
    #CH_Actuator(Cool,Heat)
    
    #AddDoorMaterial(idf,m,nm)
    #DoorActuator(doorConst)
    
    AddWinMaterial(idf,wmg,wmgs)
    wC = WindowActuator(windowType)

    AddRockwoolT(idf,m)
    AddRockwoolE(idf,m)
    AddXPS(idf,m)
    AddCement(idf,m)
    rC = RoofActuator(roofinsID,roofthickness,AreaDict)
    
    AddPlaster(idf,m)
    fC = FloorActuator(floorinsID,floorthickness,AreaDict)
    
    AddPVC(idf,m)
    AddEPST(idf,m)
    AddEPSE(idf,m)
    waC = WallActuator(wallinsID,wallcType,wallthickness,AreaDict)
    
    
    try:
        idf.run()
    except:
        with open('eplusout.err','r') as file:
            a = file.read()
            print('Error!   ' + a)
    
    
    fname = dir + '/eplustbl.htm'

    

    filehandle = open(fname, 'r').read() # get a file handle to the html file

    htables = readhtml.titletable(filehandle) # reads the tables with their titles

    #Optimization Variables
    HeatingDemand = htables[3][1][1][5]
    CoolingDemand = htables[3][1][2][4]
    DiscomfortHours = htables[-1][1][3][1]
    Cost = wC + waC + fC + rC
    #Refining
    BuildingArea = 128.32
    HeatingDemand = HeatingDemand / BuildingArea
    CoolingDemand = CoolingDemand / BuildingArea
    DiscomfortHours = DiscomfortHours / 8640 * 100
    
    O = [HeatingDemand,CoolingDemand,DiscomfortHours,Cost]
    return O
    
    
