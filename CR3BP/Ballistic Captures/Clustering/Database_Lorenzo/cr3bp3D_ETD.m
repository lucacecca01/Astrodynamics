clearvars
close all

% System selection
% 1 - Sun-Jupiter           mu = 9.54e-04
% 2 - Jupiter-Ganymede      mu = 7.80e-05
% 3 - Jupiter-Europa        mu = 2.53e-05
% 4 - Sun-Neptune           mu = 5.15e-05
% 5 - Sun-Mars              mu = 3.22e-07
% 6 - Sun-Earth             mu = 3.00e-06
% 7 - Sun-Psyche            mu = 1.15e-11
% 8 - Earth-Moon            mu = 1.22e-02
% 9 - Sun-Pluto             mu = 6.55e-09
% 10 - Pluto-Charon         mu = 1.04e-01
sys = 8 ;

G = 6.674e-20 ;             % [km^3 / (kg*s^2)]
m_1vec(1) = 1.9885e30 ;    m_2vec(1) = 1.89819e27 ;  R_refvec(1) = 778.4e6 ;    R_P_kmvec(1) = 71492 ;
m_1vec(2) = 1.89819e27 ;   m_2vec(2) = 1.48e23 ;     R_refvec(2) = 1070400 ;    R_P_kmvec(2) = 5262 ;
m_1vec(3) = 1.89819e27 ;   m_2vec(3) = 4.80e22 ;     R_refvec(3) = 670900 ;     R_P_kmvec(3) = 1561 ;
m_1vec(4) = 1.9885e30 ;    m_2vec(4) = 1.0243e26 ;    R_refvec(4) = 4498252900 ; R_P_kmvec(4) = 49528 ;
m_1vec(5) = 1.9885e30 ;    m_2vec(5) = 6.417e23 ;     R_refvec(5) = 227900000 ;  R_P_kmvec(5) = 3402.4 ;
m_1vec(6) = 1.9885e30 ;    m_2vec(6) = 398600/G ;     R_refvec(6) = 149.6e6 ;    R_P_kmvec(6) = 6378.0 ;
m_1vec(7) = 1.9885e30 ;    m_2vec(7) = 2.287e19 ;     R_refvec(7) = 4.36921e11 ; R_P_kmvec(7) = 278 ;
m_1vec(8) = 5.9724e24 ;    m_2vec(8) = 7.348e22 ;     R_refvec(8) = 384748 ;      R_P_kmvec(8) = 1737.4 ;
m_1vec(9) = 1.9885e30 ;    m_2vec(9) = 1.303e22 ;     R_refvec(9) = 5.87e9 ;       R_P_kmvec(9) = 2376.6 ;
m_1vec(10) = 1.303e22 ;    m_2vec(10) = 1.52e21 ;     R_refvec(10) = 19591.4 ;     R_P_kmvec(10) = 606 ;

mu_1 = m_1vec(sys) * G ;
mu_2 = m_2vec(sys) * G ;
R_ref = R_refvec(sys) ;
R_P = R_P_kmvec(sys) / R_ref ;

if sys == 8
    mu_1 = 3.9860043543609598e5 ;
    mu_2 = 4.9028000661637961e3 ;
    R_ref = 384399 ;
end

mu = mu_2 / (mu_1 + mu_2) ;
T = 2*pi*sqrt(R_ref^3/(mu_1 + mu_2)) ;
n = 1 / sqrt(R_ref^3/(mu_1 + mu_2)) ;
r_Hill = (mu/3)^(1/3) ;
circle_Hill = circle(1-mu, 0, r_Hill) ;

LP = lagrangePoints(mu) ;
L1 = LP(1) ;
L2 = LP(2) ;
L3 = LP(3) ;
x_L4 = LP(4,1) ;
y_L4 = LP(4,2) ;
x_L5 = LP(5,1) ;
y_L5 = LP(5,2) ;

% Jacobi constants at the Lagrange points in barycentric coordinates
x_CJL = LP(:,1) ;
y_CJL = LP(:,2) ;
CJ_L = x_CJL.^2 + y_CJL.^2 ...
    + 2*(1-mu)./sqrt((x_CJL + mu).^2 + y_CJL.^2) ...
    + 2*mu./sqrt((x_CJL - (1-mu)).^2 + y_CJL.^2) ;

%% Initial condition from GAMMA in Cartesian barycentric synodic coordinates
GAMMA = 0.14 ;
CJ = GAMMA*(CJ_L(4) - CJ_L(1)) + CJ_L(1) ;
velocity_branch = 1 ;

% Position relative to the second primary, at tau = 0
x20 = -0.2180 ;
y20 = -0.1188 ;
z20 = 0 ;

beta_deg = round(rad2deg(0.3839724)/0.5)*0.5 ;
beta = deg2rad(beta_deg) ;
r20 = sqrt(x20^2 + y20^2) ;
v20 = sqrt(2*mu/r20) ;

% The geometry used by TestSingolo, written without curvilinear states
x0_from_m1 = 1 + x20 ;
y0_from_m1 = y20 ;
r10 = sqrt(x0_from_m1^2 + y0_from_m1^2) ;
gamma = atan2(y20, -x20) ;
sigma = asin((2*(1-mu)/r10 - 2*mu*x0_from_m1 - 1 ...
    + 2*x0_from_m1 - CJ) / (2*r20*v20)) ;
alfa = [sigma - gamma, pi - sigma - gamma] ;
if alfa(2) > pi
    alfa(2) = alfa(2) - 2*pi ;
end

% Relative velocity to the second primary in inertial axes at tau = 0
xdot20 = v20*cos(alfa(velocity_branch))*cos(beta) ;
ydot20 = v20*sin(alfa(velocity_branch))*cos(beta) ;
zdot20 = v20*sin(beta) ;

% Initial state directly in the barycentric synodic frame
x0B = 1 + x20 - mu ;
y0B = y20 ;
z0B = z20 ;
xdot0B = xdot20 + y20 ;
ydot0B = ydot20 - x20 ;
zdot0B = zdot20 ;
s0l = [x0B, y0B, z0B, xdot0B, ydot0B, zdot0B] ;

%% Propagation
anni_simulazione = 2 ;
tau_span = [0, 2*pi*anni_simulazione] ;
reltol = 1e-12 ;
abstol = 1e-12 ;
options = odeset('RelTol', reltol, 'AbsTol', abstol) ;

[tau, s] = ode89(@(tau,s) cart_synB_3D(s, mu), tau_span, s0l, options) ;

x = s(:,1) ;
y = s(:,2) ;
z = s(:,3) ;
xdot = s(:,4) ;
ydot = s(:,5) ;
zdot = s(:,6) ;

r1 = sqrt((x + mu).^2 + y.^2 + z.^2) ;
r2 = sqrt((x - (1-mu)).^2 + y.^2 + z.^2) ;
CJ_history = -(xdot.^2 + ydot.^2 + zdot.^2) ...
    + x.^2 + y.^2 + 2*(1-mu)./r1 + 2*mu./r2 ;
disp(CJ_history(1))

% Position relative to the second primary
x2 = x - (1-mu) ;
y2 = y ;
z2 = z ;
d = sqrt(x2.^2 + y2.^2 + z2.^2) ;

% Inertial velocity relative to each primary
x2p = xdot - y ;
y2p = ydot + x - (1-mu) ;
z2p = zdot ;
x1p = xdot - y ;
y1p = ydot + x + mu ;
z1p = zdot ;

epsilon1 = (x1p.^2 + y1p.^2 + z1p.^2)/2 - (1-mu)./r1 ;
epsilon2 = (x2p.^2 + y2p.^2 + z2p.^2)/2 - mu./r2 ;

% Inertial coordinates relative to the first primary
xcart = (x + mu).*cos(tau) - y.*sin(tau) ;
ycart = y.*cos(tau) + (x + mu).*sin(tau) ;

tau_g = 0:0.03:2*pi ;

figure(2)
hold on
plot(0, 0, 'ok')
plot(cos(tau_g), sin(tau_g), '.', 'color', [0.6,0.6,0.6], 'linewidth', 3)
plot(cos(tau_g(1)), sin(tau_g(1)), 'dk', 'markersize', 10)
plot3(xcart, ycart, z, 'k', 'linewidth', 1)
plot3(xcart(1), ycart(1), z(1), 'ks', 'markersize', 10)
plot3(xcart(end), ycart(end), z(end), 'k*', 'markersize', 10)
xlabel('$x_2$ [LU]', 'Interpreter', 'latex', 'FontSize', 14)
ylabel('$y_2$ [LU]', 'Interpreter', 'latex', 'FontSize', 14)
zlabel('$z_2$ [LU]', 'Interpreter', 'latex', 'FontSize', 14)
set(gca, 'FontSize', 16, 'TickLabelInterpreter', 'latex')
box on
axis equal
axis([-1.2, 1.2, -1.2, 1.2])

figure(3)
hold on
plot(-mu, 0, 'ok')
plot(1-mu, 0, '*', 'color', [0.6,0.6,0.6], 'markersize', 10)
plot(circle_Hill(:,1), circle_Hill(:,2), ':k')
plot3(x, y, z, 'k', 'linewidth', 1)
plot3(x(end), y(end), z(end), 'k*', 'markersize', 10)
plot([L1,L2,L3,x_L4,x_L5], [0,0,0,y_L4,y_L5], '*', 'color', [0.6,0.6,0.6])
xlabel('$x$ [LU]', 'Interpreter', 'latex', 'FontSize', 14)
ylabel('$y$ [LU]', 'Interpreter', 'latex', 'FontSize', 14)
zlabel('$z$ [LU]', 'Interpreter', 'latex', 'FontSize', 14)
legend({'Earth', 'Moon', "Hill's sphere", 'Trajectory', ...
    'Final position', 'Lagrange points'}, ...
    'Location', 'southoutside', 'Orientation', 'horizontal')
set(gca, 'FontSize', 16, 'TickLabelInterpreter', 'latex')
box on
axis equal
axis([-1.2, 1.8, -1.5, 1.5])

figure(5)
hold on
plot(tau, epsilon2, 'k', 'linewidth', 1.5)
plot([0,tau(end)], [0,0], 'color', [0.6,0.6,0.6])
xlabel('$\tau$ [TU]', 'Interpreter', 'latex')
ylabel('$\varepsilon_2$ [VU$^2$]', 'Interpreter', 'latex')
set(gca, 'FontSize', 16, 'TickLabelInterpreter', 'latex')
box on

figure(6)
hold on
plot(tau, epsilon1, 'k', 'linewidth', 1.5)
plot([0,tau(end)], [0,0], 'color', [0.6,0.6,0.6])
xlabel('$\tau$ [TU]', 'Interpreter', 'latex')
ylabel('$\varepsilon_1$ [VU$^2$]', 'Interpreter', 'latex')
set(gca, 'FontSize', 16, 'TickLabelInterpreter', 'latex')
box on
