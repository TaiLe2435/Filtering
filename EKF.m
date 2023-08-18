% ================ Main =====================

% generating 2D data from a mobile robot with unicycle kinematics
dt = 0.02; % 180000 steps in an hour
tspan = 0:dt:4; % time for simulation
num_iterations = 100;

% ode with initial conditions 
i1 = [0 0 0];
v = 10;                     % change v and w to make square path
w = 2;

% ground truth data
[t,gt] = ode45(@(t,gt) unicycle_ode(t, gt, v, w),tspan,i1); % need to update model to simulate pauses
t = [t; t(end)+dt];

w0 = [0.05; 0.05; pi/8]; % m, m, rad

gt = gt + [randn(size(gt,1), 1)*w0(1) ... 
           randn(size(gt,1), 1)*w0(2) ...
           randn(size(gt,1), 1)*w0(3)];

Q_accel = 0.1;   % m/s^2 0.03
Q_gyro = pi/(16^2); % rad/s   pi/16

% simulated IMU measurement
z = imu(w, v, gt, dt) + [randn(size(gt,1)+1, 1)*Q_accel ... 
                         randn(size(gt,1)+1, 1)*Q_accel ...
                         randn(size(gt,1)+1, 1)*Q_gyro];

% simulating angular random walk
drift = 0;
for i=1:num_iterations
    z(i, 3) = z(i, 3) + drift;
    drift = drift + pi/360; % drift around 180deg/hr | 0.5deg/sec
end

% ================== static data ===================================
setup = [randn(size(gt,1)+1, 1)*Q_accel ... 
        randn(size(gt,1)+1, 1)*Q_accel ...
        randn(size(gt,1)+1, 1)*Q_gyro];

drift = 0;
for i=1:size(gt)
    setup(i, 3) = setup(i, 3) + drift;
    drift = drift + pi/360; % drift around 180deg/hr | 0.5deg/sec
end

% numerically differentiate to get simulated imu data
vx = diff(setup(:,1)) ./ diff(t(size(setup(:,1))));
ax = diff(vx) ./ diff(t(size(vx)));

vy = diff(setup(:,2)) ./ diff(t(size(setup(:,2))));
ay = diff(vy) ./ diff(t(size(vy)));

wz = diff(setup(:,3)) ./ diff(t(size(wz)));

% signal preprocessing
bias = zeros(size(setup, 1), 3);
bias = biasCorrection(setup(:,1), setup(:,2), wz); % param 1 and 2 need to be updated to wx and wy
wzBias = bias(1,3);
wzStatic = wz - wzBias;

% NMNI
wTh = NMNI(setup(:,1), setup(:,2), wzStatic);      % param 1 and 2 need to be updated to wx and wy

% NDZTA
aTh = NDZTA(ax, ay, setup(:,3));                   % param 3 needs to be updated to az

% logic for zero update
for i=1:length(wzStatic)
    if abs(z(i,1)) < aTh(1,1)
       z(i,1) = 0;
    end
    if abs(z(i,2)) < aTh(1,2)
        z(i,2) = 0;
    end
    if abs(z(i,3)) < wTh(1,3)
        z(i,3) = 0;
    end
end

% figure(10);
% hold on;
% plot(z);
% hold off;

% Initialize states and covariance
x0(1:2,1)=gt(1,1:2); 
x0(3,1)=0;
P0=eye(3);

ekf(dt, z, x0, P0, v, w, gt) % this z still has bias in gyro

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
        a = [0; 0];
    
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
    Hjacobian = eye(3,3);

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

% ===================== Preprocessing functions ===========================

function Bias = biasCorrection(wx, wy, wz) % Bias correction for gyro drift compensation

    wSum = zeros(1, 3);
    wSum = [sum(wx) sum(wy) sum(wz)];
    n = length(wz);

    wxBias = wSum(1,1)/n;
    wyBias = wSum(1,2)/n;
    wzBias = wSum(1,3)/n;

    Bias = [wxBias, wyBias, wzBias];

end

function wTh = NMNI(wx, wy, wz) % Signal processing for gyro
    % function should be called when object is stationary and then be
    % passed sampled data and return new thresholds

    wxTh = max(wx);
    wyTh = max(wy);
    wzTh = max(wz);

    wTh = [wxTh, wyTh, wzTh];

end

function aTh = NDZTA(ax, ay, az) % Signal processing for acc
    % function should be called when object is stationary, then it should
    % be passed sampled data and return new thresholds
   
    axTh = max(ax);
    ayTh = max(ay);
    azTh = max(az);

    aTh = [axTh, ayTh, azTh];

end

function[px, py, pz] = areaIntegration(ax, ay, az)
    dt = 0.2;

    for i=1:length(ax) 
        vx = vx_ + 0.5*(ax(i) + ax(i+1))*dt;
        vy = vy_ + 0.5*(ay(i) + ay(i+1))*dt;
        vz = vz_ + 0.5*(az(i) + az(i+1))*dt;
    end

    for j=1:length(vx)
        px = px_ + 0.5*(vx(j) + vx(j+1))*dt;
        py = py_ + 0.5*(vy(j) + vy(j+1))*dt;
        pz = pz_ + 0.5*(vz(j) + vz(j+1))*dt;
    end

end