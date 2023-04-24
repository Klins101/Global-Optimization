# Solving optimization problems 
  
# 1. Solving optimization problem using simulated Annealing

Simulated Annealing is a global optimization algorithm inspired by the annealing process in metallurgy where metals are heated to a high temperature and cooled slowly in a controlled manner.  Simulated annealing works in a similar way. The algorithm starts with a random solution to the problem. It then repeatedly generates new solutions, and accepts or rejects them based on a probability. The probability of accepting a worse solution decreases as the temperature decreases. The algorithm terminates when the temperature reaches a certain level, or when a certain number of iterations have been performed.

$$
f(x)=0.3x_1^3 -1.5x_2^2 +7.5x_1
$$
## **Function visualization**

```matlab:Code
clear;clc;
dim = 50;
x1 = linspace( -10, 10, dim); 
x2 = linspace(-10, 10, dim);
[X,Y] = meshgrid(x1,x2);

ff = @(x1, x2) 0.3*x1.^3 - 1.5*x2.^2 + 7.5*x1 ; 
figure;
surf(X,Y,ff(X,Y))
xlabel('x(1)')
ylabel('x(2)')
zlabel('f')
grid on;
```

![figure_0.png](README_images/figure_0.png)

## Performing optimization

```matlab:Code
% optimizing the booth's function using simulated annealing 
ff = @(x) 0.3*x(1).^3 - 1.5*x(2).^2 + 7.5*x(1) ;
x0 = [0 0]; % initial parameters
Lb = -10*ones(1,2); % lower bound
Ub =  10*ones(1,2); % upper bound
opt = optimoptions(@simulannealbnd,...
    'Display', 'iter',... % Level of display
    'reannealinterval', 100,... % Reannealing interval
    'plotfcn', @saplotbestf);
[xf,fval,exitflag] = simulannealbnd(ff,x0,Lb,Ub,opt);
```

```text:Output
                           Best        Current           Mean
Iteration   f-count         f(x)         f(x)         temperature
     0          1              0              0            100
    10         11       -272.071       -272.071          56.88
    20         21       -496.487       -496.487        34.0562
    30         31       -518.223       -518.223        20.3907
    40         41       -519.204        -514.25        12.2087
    50         51       -524.964       -520.944        7.30977
    60         61       -524.964       -522.843        4.37663
    70         71       -524.964       -524.788        2.62045
    80         81       -524.964        -524.31        1.56896
    90         91       -524.999       -524.999       0.939395
   100        101           -525           -525        0.56245
   110        111           -525           -525        0.33676
   120        121           -525           -525       0.201631
   130        131           -525           -525       0.120724
   140        141           -525           -525      0.0722817
   150        151           -525           -525      0.0432777
   160        161           -525       -524.995       0.025912
   170        171           -525       -524.995      0.0155145
   180        181           -525       -524.999     0.00928908
   190        191           -525           -525     0.00556171
   200        201           -525           -525        0.00333
   210        211           -525       -524.999      0.0019938
   220        221           -525           -525     0.00119376
   230        231           -525           -525    0.000714748
   240        241           -525           -525    0.000427946
   250        251           -525           -525    0.000256227
   260        261           -525           -525    0.000153413
   270        271           -525           -525    9.18538e-05
   280        281           -525           -525    5.49963e-05
   290        291           -525           -525    3.29283e-05
   300        301           -525           -525    1.97154e-05

                           Best        Current           Mean
Iteration   f-count         f(x)         f(x)         temperature
   310        311           -525           -525    1.18043e-05
   320        321           -525           -525    7.06769e-06
   330        331           -525           -525    4.23169e-06
   340        341           -525           -525    2.53367e-06
   350        351           -525           -525      1.517e-06
   360        361           -525           -525    9.08284e-07
   370        371           -525           -525    5.43823e-07
   380        381           -525           -525    3.25607e-07
*  389        392           -525           -525        34.8809
   390        393           -525           -525        33.1368
   400        403           -525           -525        19.8402
   410        413           -525       -506.166        11.8791
   420        423           -525       -524.701        7.11245
   430        433           -525        -524.74        4.25849
   440        443           -525       -524.925        2.54971
   450        453           -525       -524.998        1.52661
   460        463           -525       -524.999       0.914036
   470        473           -525           -525       0.547267
   480        483           -525           -525       0.327669
   490        493           -525           -525       0.196188
   500        503           -525           -525       0.117465
   510        513           -525           -525      0.0703305
   520        523           -525           -525      0.0421095
   530        533           -525           -525      0.0252125
   540        543           -525           -525      0.0150956
   550        553           -525           -525     0.00903832
   560        563           -525           -525     0.00541158
   570        573           -525           -525     0.00324011
   580        583           -525           -525     0.00193997
   590        593           -525           -525     0.00116153
   600        603           -525       -524.998    0.000695454

                           Best        Current           Mean
Iteration   f-count         f(x)         f(x)         temperature
   610        613           -525       -524.999    0.000416394
   620        623           -525           -525     0.00024931
   630        633           -525           -525    0.000149271
   640        643           -525           -525    8.93742e-05
   650        653           -525           -525    5.35117e-05
   660        663           -525           -525    3.20394e-05
   670        673           -525           -525    1.91832e-05
   680        683           -525           -525    1.14857e-05
   690        693           -525           -525     6.8769e-06
   700        703           -525           -525    4.11745e-06
   710        713           -525           -525    2.46527e-06
   720        723           -525           -525    1.47605e-06
   730        733           -525           -525    8.83765e-07
   740        743           -525           -525    5.29143e-07
   750        753           -525           -525    3.16817e-07
   760        763           -525           -525     1.8969e-07
*  763        768           -525           -525        34.4681
   770        775           -525           -525        24.0704
   780        785           -525           -525        14.4118
   790        795           -525           -525         8.6289
   800        805           -525           -525        5.16644
   810        815           -525           -525        3.09334
   820        825           -525           -525         1.8521
   830        835           -525           -525        1.10892
   840        845           -525           -525        0.66395
   850        855           -525           -525       0.397532
   860        865           -525           -525       0.238017
   870        875           -525           -525       0.142509
   880        885           -525           -525      0.0853257
   890        895           -525           -525      0.0510876
   900        905           -525       -524.934      0.0305881

                           Best        Current           Mean
Iteration   f-count         f(x)         f(x)         temperature
   910        915           -525        -524.99      0.0183142
   920        925           -525       -524.995      0.0109654
   930        935           -525           -525     0.00656538
   940        945           -525           -525     0.00393094
   950        955           -525           -525      0.0023536
   960        965           -525           -525     0.00140919
   970        975           -525           -525    0.000843732
   980        985           -525           -525    0.000505173
   990        995           -525           -525    0.000302466
  1000       1005           -525           -525    0.000181097
  1010       1015           -525           -525     0.00010843
  1020       1025           -525           -525    6.49209e-05
  1030       1035           -525           -525    3.88705e-05
  1040       1045           -525           -525    2.32732e-05
  1050       1055           -525           -525    1.39345e-05
  1060       1065           -525           -525    8.34312e-06
  1070       1075           -525           -525    4.99534e-06
  1080       1085           -525           -525    2.99089e-06
  1090       1095           -525           -525    1.79076e-06
  1100       1105           -525           -525    1.07219e-06
  1110       1115           -525           -525    6.41962e-07
```

![figure_1.png](README_images/figure_1.png)

```text:Output
Optimization terminated: change in best function value less than options.FunctionTolerance.
```

## Plot after optimization

```matlab:Code
dim = 50;
x1 = linspace( -10, 10, dim); 
x2 = linspace(-10, 10, dim);
[X,Y] = meshgrid(x1,x2);

ff = @(x1, x2) 0.3*x1.^3 - 1.5*x2.^2 + 7.5*x1 ; 
figure;
surf(X,Y,ff(X,Y))
hold on;
xlabel('x(1)')
ylabel('x(2)')
zlabel('f')
grid on;
plot3(xf(1),xf(2),fval, 'r^');
hold off;
```

![figure_2.png](README_images/figure_2.png)

## **Results after optimization**

```matlab:Code
display(xf)
```

```text:Output
xf = 1x2    
   -10   -10

```

# 2. Solving optimization problem using genetic algorithm

A genetic algorithm (GA) is a search heuristic that is routinely used to generate useful solutions to optimization and search problems. Genetic algorithms are inspired by the process of natural selection that governs the evolution of biological populations.

A GA works by creating a population of possible solutions to a problem, and then iteratively modifying that population to improve the quality of the solutions. The GA uses a process of selection, crossover, and mutation to create new solutions from the existing population.

$$
f(x)=10.2x_1^4 -3.8x_2^2 +7.1x_1
$$
## **Function visualization**

```matlab:Code
clear;clc;
dim = 50;
x1 = linspace( -10, 10, dim); 
x2 = linspace(-10, 10, dim);
[X,Y] = meshgrid(x1,x2);

ff = @(x1, x2) 10.2*x1.^4 - 3.8*x2.^2 + 7.1*x1 ; 
figure;
surf(X,Y,ff(X,Y))
xlabel('x(1)')
ylabel('x(2)')
zlabel('f')
grid on;
```

![figure_3.png](README_images/figure_3.png)

## Performing optimization with pattern search

```matlab:Code
lb = -10*ones(1,2); % lower bound
ub =  10*ones(1,2); % upper bound
A = [];
b = [];
Aeq = [];
beq = [];
ff = @(x) 10.2*x(1).^4 - 3.8*x(2).^2 + 7.1*x(1) ;
% initial parameters
x0 = [1,-5];
x = patternsearch(ff,x0,A,b,Aeq,beq,lb,ub)
```

```text:Output
Optimization terminated: mesh size less than options.MeshTolerance.
x = 1x2    
   -0.5583  -10.0000

```

## **Results after optimization**

```matlab:Code
display(x)
```

```text:Output
x = 1x2    
   -0.5583  -10.0000

```

## **Plot after optimization **

```matlab:Code
dim = 50;
x1 = linspace( -10, 10, dim); 
x2 = linspace(-10, 10, dim);
[X,Y] = meshgrid(x1,x2);
ff = @(x1, x2) 10.2*x1.^4 - 3.8*x2.^2 + 7.1*x1 ; 
figure;
surf(X,Y,ff(X,Y))
hold on;
xlabel('x(1)')
ylabel('x(2)')
zlabel('f')
grid on;
plot3(x(1),x(2),ff(x(1),x(2)), 'r^');
hold off;
```

![figure_4.png](README_images/figure_4.png)

# 3. Solving optimization problem using Particle swarm

Particle swarm is a population-based optimization algorithm that is inspired by the social behavior of birds or fish. It is a simple and efficient algorithm that can be used to solve a variety of optimization problems.

This algorithm works by having a population of particles (candidate solutions) that move around in a search space. At each step, each particle is attracted to its own best-known position (pbest) and to the global best position (gbest) that has been found so far by any particle in the swarm. The particles move towards these best positions by updating their velocities.

$$
f(x)=(x_1 -7.9)(x_2 +8.8)^3
$$
## **Function visualization**

```matlab:Code
clear;clc;
dim = 50;
x1 = linspace( -10, 10, dim); 
x2 = linspace(-10, 10, dim);
[X,Y] = meshgrid(x1,x2);

ff = @(x1, x2) (x1  - 7.9).*(x2 + 8.8).^3; 
figure;
surf(X,Y,ff(X,Y))
xlabel('x(1)')
ylabel('x(2)')
zlabel('f')
grid on;
```

![figure_5.png](README_images/figure_5.png)

## Performing optimization with particle swarm

```matlab:Code
lb = -10*ones(1,2); % lower bound
ub =  10*ones(1,2); % upper bound
ff = @(x) (x(1) - 7.9).*(x(2) + 8.8).^3;
options = optimoptions('particleswarm','SwarmSize',50,'HybridFcn',@fmincon);
rng default  % For reproducibility
nvars = 2;
[x,fval,exitflag,output] = particleswarm(ff,nvars,lb,ub,options)
```

```text:Output
Optimization ended: relative change in the objective value 
over the last OPTIONS.MaxStallIterations iterations is less than OPTIONS.FunctionTolerance.
x = 1x2    
   -10    10

fval = -1.1894e+05
exitflag = 1
output = 
      rngstate: [1x1 struct]
    iterations: 21
     funccount: 1115
       message: 'Optimization ended: relative change in the objective value ↵over the last OPTIONS.MaxStallIterations iterations is less than OPTIONS.FunctionTolerance.↵FMINCON: Local minimum found that satisfies the constraints.↵↵Optimization completed because the objective function is non-decreasing in ↵feasible directions, to within the value of the optimality tolerance,↵and constraints are satisfied to within the value of the constraint tolerance.↵↵<stopping criteria details>↵↵Optimization completed: The relative first-order optimality measure, 7.333395e-08,↵is less than options.OptimalityTolerance = 1.000000e-06, and the relative maximum constraint↵violation, 0.000000e+00, is less than options.ConstraintTolerance = 1.000000e-06.↵'
    hybridflag: 1

```

## **Results after optimization**

```matlab:Code
display(x)
```

```text:Output
x = 1x2    
   -10    10

```

## **Plot after optimization **

```matlab:Code
dim = 50;
x1 = linspace( -10, 10, dim); 
x2 = linspace(-10, 10, dim);
[X,Y] = meshgrid(x1,x2);
ff = @(x1, x2) (x1  - 7.9).*(x2 + 8.8).^3; 
figure;
surf(X,Y,ff(X,Y))
hold on;
xlabel('x(1)')
ylabel('x(2)')
zlabel('f')
grid on;
plot3(x(1),x(2),fval, 'r^');
hold off;
```

![figure_6.png](README_images/figure_6.png)

# 4. Solving optimization problem using surrogate optimization

The surrogate model is then used to guide the search for an optimal solution, without the need to evaluate the expensive objective function at every iteration.

Surrogate optimization is often used in cases where the objective function is expensive to evaluate, such as when it requires a simulation or a physical experiment. In these cases, surrogate optimization can be used to reduce the number of expensive evaluations required to find an optimal solution.

$$
f(x)=(x_1 -5.8)^2 (x_2 -4.1)^2
$$
## **Function visualization**

```matlab:Code
clear;clc;
dim = 50;
x1 = linspace( -10, 10, dim); 
x2 = linspace(-10, 10, dim);
[X,Y] = meshgrid(x1,x2);

ff = @(x1, x2) (x1  - 5.8).^2.*(x2 - 4.1).^2; 
figure;
surf(X,Y,ff(X,Y))
xlabel('x(1)')
ylabel('x(2)')
zlabel('f')
grid on;
```

![figure_7.png](README_images/figure_7.png)

## Performing optimization with surrogate optimization

```matlab:Code
rng default  % For reproducibility
lb = -10*ones(1,2); % lower bound
ub =  10*ones(1,2); % upper bound
ff = @(x) (x(1)  - 5.8).^2.*(x(2) - 4.1).^2; 
x = surrogateopt(ff,lb,ub)
```

![figure_8.png](README_images/figure_8.png)

```text:Output
Warning: Some output might be missing due to a network interruption. To get the missing output, rerun the script.
```

## **Plot after optimization **

```matlab:Code
dim = 50;
x1 = linspace( -10, 10, dim); 
x2 = linspace(-10, 10, dim);
[X,Y] = meshgrid(x1,x2);
ff = @(x1, x2) (x1  - 5.8).^2.*(x2 - 4.1).^2; 
figure;
surf(X,Y,ff(X,Y))
hold on;
xlabel('x(1)')
ylabel('x(2)')
zlabel('f')
grid on;
plot3(x(1),x(2),ff(x(1),x(2)), 'r^');
hold off;
```

![figure_9.png](README_images/figure_9.png)

# 5. Solving optimization problem using Global search 

Global search algorithm works by exploring the search space in a random or systematic way. They may use a variety of techniques to guide the search, such as local search, hill climbing. The goal of global search is to find the global optimum, which is the point in the search space where the objective function has its lowest value.

$$
f(x)=0.5x_1^4 -9.4x_2^2 +2.5
$$
## **Function visualization**

```matlab:Code
dim = 50;
x1 = linspace( -10, 10, dim); 
x2 = linspace(-10, 10, dim);
[X,Y] = meshgrid(x1,x2);

ff = @(x1, x2) 0.5*x1.^4 - 9.4*x2.^2 + 2.5; 
figure;
surf(X,Y,ff(X,Y))
xlabel('x(1)')
ylabel('x(2)')
zlabel('f')
grid on;
```

![figure_10.png](README_images/figure_10.png)

  
## Optimization of function

```matlab:Code
rng default % For reproducibility
gs = GlobalSearch;
lb = -10*ones(1,2); % lower bound
ub =  10*ones(1,2); % upper bound
x0 = [0, 0];
ff = @(x) 0.5*x(1).^4 - 9.4*x(2).^2 + 2.5 ; 
problem = createOptimProblem('fmincon','x0',x0,...
    'objective',ff,'lb',lb,'ub',ub);
[x, fval] = run(gs,problem);
```

```text:Output
GlobalSearch stopped because it analyzed all the trial points.

All 10 local solver runs converged with a positive local solver exit flag.
```

## Results after optimization

```matlab:Code
display(x)
```

```text:Output
x = 1x2    
   -0.0359   10.0000

```

## **Plot after optimization **

```matlab:Code
dim = 50;
x1 = linspace( -10, 10, dim); 
x2 = linspace(-10, 10, dim);
[X,Y] = meshgrid(x1,x2);
ff = @(x1, x2) 0.5*x1.^4 - 9.4*x2.^2 + 2.5; 
figure;
surf(X,Y,ff(X,Y))
hold on;
xlabel('x(1)')
ylabel('x(2)')
zlabel('f')
grid on;
plot3(x(1),x(2),fval, 'r^');
hold off;
```

![figure_11.png](README_images/figure_11.png)
