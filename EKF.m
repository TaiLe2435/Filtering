% ================ Main =====================

% generating 2D data from a mobile robot with unicycle kinematics
dt = 0.02;
tspan = 0:dt:4; % time for simulation
num_iterations = 100;

% ode with initial conditions 
i1 = [0 0 0];
v = 10; 
w = 2;

% ground truth data
[t,gt] = ode45(@(t,gt) unicycle_ode(t, gt, v, w),tspan,i1);

w0 = [0.05; 0.05; pi/8]; % m, m, rad

gt = gt + [randn(size(gt,1), 1)*w0(1) ... 
           randn(size(gt,1), 1)*w0(2) ...
           randn(size(gt,1), 1)*w0(3)];

Q_accel = 0.1;   % m/s^2 0.03
Q_gyro = pi/8; % rad/s   pi/16

% simulated IMU measurement
z = imu(w, v, gt, dt) + [randn(size(gt,1)+1, 1)*Q_accel ... 
                         randn(size(gt,1)+1, 1)*Q_accel ...
                         randn(size(gt,1)+1, 1)*Q_gyro];

% Initialize states and covariance
x0(1:2,1)=gt(1,1:2); 
x0(3,1)=0;
P0=eye(3);

ekf(dt, z, x0, P0, v, w, gt)

% ===================== Functions ============================

function Xdot = unicycle_ode(t,f,v,w) % just to get the simulated ground truth
    % Process model
    Xdot = zeros(3,1);

    Xdot(1) = v*cos(f(3));
    Xdot(2) = v*sin(f(3));
    Xdot(3) = w; % usually w but try function of time sin(0.5*t)
end

function f = unicycle(f, v, w, dt)
    % Process model for KF
    f(1,1) = f(1,1) + v*cos(f(1,3))*dt;
    f(1,2) = f(1,2) + v*sin(f(1,3))*dt;
    f(1,3) = f(1,3) + w*dt;

end

function Fk = unicycle_linearized(f,v, dt)
    % Jacobian of process model
    Fk = [  1 0 -v*sin(f(1,3))*dt;
            0 1  v*cos(f(1,3))*dt;
            0 0         1];
end

function IMUsim = imu(w, v, gt, dt)

    % Initialize IMU data containers
    pos = zeros(2, 1);
    head = 0;
    positionStore = zeros(1,2);
    headStore = zeros(1);

    for i=1:length(gt)
        % calculate heading from gyro
        orientationChange = w * dt;
        head = head + orientationChange;
        
        headStore = [headStore; head];
    
        % convert input velocities to acceleration
        a = [0 ;0];
    
        % transform acceleration data to global coordinates
        Rz = [cos(head) -sin(head);
              sin(head)  cos(head)];
        aT = Rz * a; % 2x2 ' 2x1
    
        % already have velocity from kinematic model inputs
        % don't feel like going to acc then back to vel
        % so transform vel to global coordinates instead
        vel = [v*cos(head); v*sin(head)];
    %     vT = Rz * vel;
    
        positionChange = vel * dt;
        pos = pos + positionChange;
    
        positionStore = [positionStore; 
                        pos(1), pos(2)];
    end

    IMUsim = [positionStore(:,1), positionStore(:,2), headStore];
end

function Hjacobian = imu_linearized(x)
    
    % not correct for the actual model, but in simulation the values are
    % directly accesible I guess?
    Hjacobian = eye(3,3); % doesn't seem right

end

function [xk_, P_] = predict(xk, P, Qk, f, Fk)
    % States
    xk_ = f(xk);

    % Covariance
    P_ = Fk*P*Fk' + Qk;
end

function [xk, P] = update(xk_, P_, R, z, h, Hjacob)
    % get size of states
    [~,n] = size(xk_);
    
    % handle Jacobian of measurement model
    Hk = Hjacob;

    % Innovation Covariance
    S = Hk*P_*Hk' + R;

    % Measurement Model
    zh = h;

    % Innovation
    innovation = z - zh;

    % Kalman Gain
    K = P_*Hk' / (S);

    % Update states and covariance
    xk = xk_' + K*innovation';
    P = (eye(n) - K*Hk) * P_;

end

function ekf(dt, z, x0, P0, v, w, gt)
    
    n = size(x0, 1);
    m = size(z, 1);
    T = size(z, 1);

    xk = zeros(T,n);
    xk_ = xk;
    P = zeros(n,n,T);
    P_ = P;
    zModel = zeros(T,n);

    f = @(x) unicycle(x, v, w, dt);

    Fk = @(x) unicycle_linearized(x, v, dt);

    Wk = @(x,dt) [cos(x(3))*dt 0;
                  sin(x(3))*dt 0;
                      0       dt];

    Qk = [0.05 0;
          0 0.05];

%     Q = Wk*Qk*Wk'; 

    Q = @(x,dt) [cos(x(3))*dt 0; sin(x(3))*dt 0; 0 dt]*...
            [0.05 0; 
             0 pi/8]*[cos(x(3))*dt 0; sin(x(3))*dt 0; 0 dt]';

    h = imu(w, v, gt, dt);

    Hjacobian = imu_linearized;

    Rk = [0.5 0 0;         % assuming 5 cm slip
          0 0.5 0;
          0  0 pi/4];       % about 20 deg slip

    k0=1;
    kF=T;

    xk_(k0,:) = x0;
    P_(:,:,k0) = P0;

    for k=k0:kF
        [xk(k,:), P(:, :, k, 1)] = update(xk_(k,:), P_(:,:,k,1), Rk, z(k,:), h(k,:), Hjacobian);

        [xk_(k+1,:), P_(:, :, k+1, 1)] = predict(xk(k,:), P(:,:,k,1), Q(xk(k,:),dt), f, Fk(xk(k,:)));

        zModel(k,:) = h(k,:); % to get measurement model output 
                              % but what is the input to calculate
                              % measurement model values?
                              % should they be theoretical values given
                              % by control inputs?
    end

show_plots(gt, z, xk, zModel);    

end

function show_plots(gt, z, xk, zModel)
    
    figure(1);
    hold on;
    plot(xk(:,1), xk(:, 2));
    plot(gt(:,1), gt(:, 2));
    legend('estimate', 'ground truth');
    xlim([-inf inf])
    ylim([-inf inf])
    hold off;

    figure(2);
    hold on;
    plot(z(:,1), z(:, 2));
    plot(zModel(:,1), zModel(:, 2));    
    legend('Measurement', 'Measurement Model');
    xlim([-inf inf])
    ylim([-inf inf])
    hold off;

end