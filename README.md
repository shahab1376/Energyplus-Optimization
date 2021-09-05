# Energyplus-optimization
Optimization of energy parameters for a house via material selection.

We've worked on a house with the map below:

![Map of House](https://raw.githubusercontent.com/shahab1376/Energyplus-optimization/master/map.png)

Our work was on these 4 rooms.
The software we have worked with is energyplus. Download is for free and we've used 8.3 version, but I don't think higher versions will give error. It can be easily called and be changed in eppy environment. The file named model is the same as the original file. The PDF called model is actually the code space of this original file.

Options to apply for optimization:
1- Cooling and heating temperature
2- Floor insulation
3- Roof insulation
4- Wall insulation
5- Windows
6- Doors
7- Lighting
All of these individual combinations must be analysed to find which temperature with which insulation with which window and which door selection is optimized.
We have 4 goals that should be minimized:

1,2. Heating demand (KWh/m^2a) and Cooling demand (KWh/m^2a).

Report: Annual Building Utility Performance Summary

3. Discomfort hours (%).

 Comfort and Setpoint Not Met Summary

4. Investment Cost (Rials).

We have used eppy environment to define our objective function to be optimized. You can install it by typing the command below in your terminal.

```
pip install eppy
```

Or you can define the exact version we used.

```
pip install eppy==8.3
```

Multi-objective grey wolf optimization (MOGWO) and NSGAII algorithms are used in a comparative approach to perform optimization.

For NSGAII we have used platypus. Install it as below.

```
pip install platypus
```

