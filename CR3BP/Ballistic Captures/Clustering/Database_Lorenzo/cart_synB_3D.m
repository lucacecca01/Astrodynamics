function outputdot = cart_synB_3D(s,mu)

%  Data

x = s (1) ;
y = s (2) ;
z = s (3) ;
xdot = s (4) ;
ydot = s (5) ;
zdot = s (6) ;

%  Compute acceleration

r1 = sqrt( (x+mu)^2 + y^2 + z^2 ) ;

r2 = sqrt( (x-(1-mu))^2 +y^2 + z^2 ) ;

% omega = 0.5*(x^2+y^2) + (1-mu)/r1 + mu/r2 ;

xacc = 2*ydot + x - (1-mu)*(x+mu)/r1^3 - mu*(x-(1-mu))/r2^3 ;
    
yacc = -2*xdot + y - (1-mu)*y/r1^3 - mu*y/r2^3 ;

zacc = - (1-mu)*z/r1^3 - mu*z/r2^3 ;

%%  Compose derivative

outputdot = [xdot; ydot; zdot; xacc; yacc; zacc] ;