% Testing KF2

% generating 2D data from a mobile robot with unicycle kinematics
dt = 0.02; % 180000 steps in an hour
tspan = 0:dt:4; % time for simulation
num_iterations = 100;

% ode with initial conditions 
i1 = [0 0 0];               % x, y, theta
v = 10;                     % change v and w to make square path
w = 2;

% ground truth data
[t,gt] = ode45(@(t,gt) unicycle_ode(t, gt, v, w),tspan,i1);
t = [t; t(end)+dt];

w0 = [0.05; 0.05; pi/4]; % m, m, rad

% gt = gt + [randn(size(gt,1), 1)*w0(1) ... 
%            randn(size(gt,1), 1)*w0(2) ...
%            randn(size(gt,1), 1)*w0(3)];

Q_accel = 0.05;   % m/s^2 0.03
Q_gyro = pi/(16^2); % rad/s   pi/16

% simulated measurements v and heading error
z = measurement(w, v, gt, dt) + [randn(size(gt,1)+1, 1)*Q_accel ... 
                                 randn(size(gt,1)+1, 1)*Q_accel ...
                                 randn(size(gt,1)+1, 1)*Q_accel ...
                                 randn(size(gt,1)+1, 1)*Q_gyro];

% used for getting rpy and acc
robotStates = imu(w, v, gt, dt) + [randn(size(gt,1)+1, 1)*Q_accel ... 
                                   randn(size(gt,1)+1, 1)*Q_accel ...
                                   randn(size(gt,1)+1, 1)*Q_gyro];

vx = diff(robotStates(:,1)) ./ diff(t);
vy = diff(robotStates(:,2)) ./ diff(t);

vx = [0; vx];
vy = [0; vy];

ax = diff(vx) ./ diff(t);
ay = diff(vy) ./ diff(t);

ax = [0; ax];
ay = [0; ay];

acc.x = ax/10000; % for some reason values are off by 10,000?
acc.y = ay/10000;
acc.z = zeros(length(ax));

yaw = robotStates(:,3);

% z(:,3) = wrapToPi(z(:,3));

% Initialize states and covariance
% x0(1:2,1)=gt(1,1:2); 
% x0(3,1)=0;
x0 = zeros(12,1);
P0 = eye(12);

foo = ekf(dt, z, x0, P0, v, w, gt, acc, yaw, robotStates);

% ===================== Functions ============================

function Xdot = unicycle_ode(t,f,v,w) % just to get the simulated ground truth
    % Process model
    Xdot = zeros(3,1);

    Xdot(1) = v*cos(f(3));
    Xdot(2) = v*sin(f(3));
    Xdot(3) = w; % usually w but try function of time sin(0.5*t)
end

function f = unicycle(f, v, w, dt) % don't really need
    % Process model for KF
    f(1,1) = f(1,1) + v*cos(f(1,3))*dt;
    f(1,2) = f(1,2) + v*sin(f(1,3))*dt;
    f(1,3) = f(1,3) + w*dt;

end

function Fk = PDR_linearized(a, roll, pitch, yaw, dt) % need way to access rpy
    % Jacobian of process model
    Fk = [];

    betaG = 0.2;
    betaA = 0.2;

    I33 = eye(3);
    zero33 = zeros(3,3);

%     roll = phi.x;
%     pitch = phi.y;
%     yaw = phi.z;

    for i=1:length(yaw)
        cR = cos(roll);
        sR = sin(roll);
        cP = cos(pitch);
        sP = sin(pitch);
        cY = cos(yaw(i));
        sY = sin(yaw(i));
    
        C = [cP*cY, sP*sR*cY-cR*sY, cR*sP*cY+sR*sY;
             cP*sY, sR*sP*sY+cR*cY, cR*sP*sY-cY*sR;
              -sP,       sR*cP,           cR*cP];
    
        S = [0 -a.z(i) a.y(i);
             a.z(i) 0 -a.x(i);
             -a.y(i) a.x(i) 0];
        
    
        Fk(:,:,i) = [zero33 zero33    -C      zero33;
                S  zero33   zero33      C;
              zero33 zero33 -betaG*I33   zero33;
              zero33 zero33   zero33   -betaA*I33];
    end


end

function Gk = noise_linearized(roll, pitch, yaw)

%     roll = phi.x;
%     pitch = phi.y;
%     yaw = phi.z;

    for i=1:length(yaw)
    cR = cos(roll);
    sR = sin(roll);
    cP = cos(pitch);
    sP = sin(pitch);
    cY = cos(yaw(i));
    sY = sin(yaw(i));

    C = [cP*cY sP*sR*cY-cR*sY cR*sP*cY+sR*sY;
         cP*sY sR*sP*sY+cR*cY cR*sP*sY-cY*sR;
          -sP       sR*cP           cR*cP];

    I33 = eye(3);
    zero33 = zeros(3,3);

    Gk(:,:,i) = [-C zero33 zero33 zero33;
                zero33 C zero33 zero33;
                zero33 zero33 I33 zero33;
                zero33 zero33 zero33 I33];
    end
end

function MeasurementModel = measurement(w, v, gt, dt)

    % Get the KF1 data
    KF1 = readmatrix("KF1Data.csv");
    heading_error = KF1(:,2) - KF1(:,1);

    % Initialize IMU data containers
    pos = zeros(2, 1);
    head = 0;
    positionStore = zeros(1,2);
    headStore = zeros(1);
%     vel = zeros(1, 3);
    vel = [10, 0, 0];

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
        vel = [vel; v*cos(head) v*sin(head) 0]; % make 3D
    %     vT = Rz * vel;
    
%         positionChange = vel * dt;
%         pos = pos + positionChange;
    
%         positionStore = [positionStore; 
%                         pos(1), pos(2)];
    end

    MeasurementModel = [heading_error, vel(:,1), vel(:,2), vel(:,3)]; % change to vel
end

function IMU = imu(w, v, gt, dt)

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

    IMU = [positionStore(:,1), positionStore(:,2), headStore];
end

function Hjacobian = Meas_linearized(roll, pitch, yaw) % phi % maybe have to switch back to I since my measurements are already mapped to states I desire
    
    zero13 = zeros(1,3);
    zero33 = zeros(3,3);
    I33 = eye(3);

%     roll = phi.x;           % figure out how to access rpy in this function
%     pitch = phi.y;
%     yaw = phi.z;

for i=1:length(yaw)
    tR = tan(roll);
    cY = cos(yaw(i));
    sY = sin(yaw(i));

    Hjacobian(:,:,i) = [tR*cY tR*sY -1 zero13 zero13 zero13;
                zero33  I33   zero33 zero33];
end
end

function [xk_, P_] = predict(xk, P, Qk, xk_prev, Fk, Gk)
    % States
    xk_ = xk' + Fk*(xk' - xk_prev'); % taylor expansion | might be incorrect

    % Covariance
    P_ = Fk*P*Fk' + Gk*Qk*Gk; % Jacobian
end

function [xk, P] = update(xk_, P_, R, z, h, Hjacob)
    % get size of states
    [~,n] = size(xk_);
    
    % handle Jacobian of measurement model
    Hk = Hjacob; % Jacobian

    % Innovation Covariance
    S = Hk*P_*Hk' + R;

    % Measurement Model
    zh = h; % Taylor expansion

    % Innovation
    innovation = z - zh;

    % Kalman Gain
    K = P_*Hk' / (S);

    % Update states and covariance
    xk = xk_' + K*innovation';
    P = (eye(n) - K*Hk) * P_;

end

function xk = ekf(dt, z, x0, P0, v, w, gt, acc, yaw, states)
    
    n = size(x0, 1);
    m = size(z);
    T = size(z, 1);

    xk = zeros(T,n);
    xk_ = xk;
    P = zeros(n,n,T);
    P_ = P;
    zModel = zeros(m);

    Fk = PDR_linearized(acc, 0, 0, yaw, dt);

    Gk = noise_linearized(0, 0, yaw);
                                                        % Q artificially smears new normal distr
    Q = eye(12);                                        % increasing the uncertainty of prediction
    Q(1,1) = pi/16; Q(2,2) = pi/16; Q(3,3) = pi/16;     % larger Q means that the model does a 
    Q(4,4) = 0.1; Q(5,5) = 0.1; Q(6,6) = 0.1;           % bad job predicting the process and
    Q(7,7) = 0.1; Q(8,8) = 0.1; Q(9,9) = 0.1;           % we have to extend the range that we 
    Q(10,10) = 0.1; Q(11,11) = 0.1; Q(12,12) = 0.1;     % think the state vector lies in

    h = measurement(w, v, gt, dt);

    Hjacobian = Meas_linearized(0, 0, yaw);

    Rk = [(pi/16)^2 0 0 0;         % larger R means ignore measurements and rely on prediction
          0 (0.6)^2 0 0;           % smaller R means rely more on measurements
          0 0 (0.6)^2 0;
          0 0 0 (0.6)^2];       

    k0=1;
    kF=T;

    xk_(k0,:) = x0;
    P_(:,:,k0) = P0;

    for k=k0:kF
        [xk(k,:), P(:, :, k, 1)] = update(xk_(k,:), P_(:,:,k,1), Rk, z(k,:), h(k,:), Hjacobian(:,:,k));

        [xk_(k+1,:), P_(:, :, k+1, 1)] = predict(xk(k,:), P(:,:,k,1), Q, xk_(k,:), Fk(:,:,k), Gk(:,:,k)); % change f to x_prev

        zModel(k,:) = h(k,:); 
    end
    
%     show_plots(gt, z, xk, zModel);    

   
    
    tempGt = [0; gt(:,3)];

    lenGt = size(tempGt);
    lenXk = size(xk(:,3));

    temp = states(:,3) - xk(:,3);

    figure(11);
    hold on;
    title('Heading');
    plot(temp, '--', 'LineWidth', 2);
    plot(tempGt);
%     plot(xk(:,3));
    legend('Estimated Heading', 'Ground Truth');
    xlabel('time (s)');
    ylabel('heading (rad)');
    hold off;

%     figure(12);
%     hold on;
%     plot(xk(:,3));
%     plot(xk(:,4));
%     plot(xk(:,5));  
%     hold off;

    figure(13);
    hold on;
    title('Velocity on x-axis');
    vx = z(:,2) - xk(:,4);
    vy = z(:,3) - xk(:,5);
    plot(vx, '--', 'LineWidth', 2);
%     plot(z(:,2));
    plot(zModel(:,2));
    legend('Vx', 'Ground Truth Vx');
    xlabel('time (s)');
    ylabel('velocity (m/s^2)');
    hold off;

    figure(14);
    hold on;
    title('Velocity on y-axis');
    plot(vy, '--', 'LineWidth', 2);
%     plot(z(:,3));
    plot(zModel(:,3));
    legend('Vy', 'Ground Truth Vy');
    xlabel('time (s)');
    ylabel('velocity (m/s^2)');
    hold off;
    
    pos = zeros(1,2);
    store = zeros(1,2);
    for i=1:length(vx)
        deltaPx = vx(i) * dt;
        deltaPy = vy(i) * dt;
        change = [deltaPx, deltaPy];

        pos = pos + change;
        store = [store; pos(1), pos(2)];
    end

    figure(15);
    hold on;
    title('Mobile Robot EKF', 'FontSize', 14);
    plot(store(:,1),store(:,2), '--', 'LineWidth', 2);
    plot(gt(:,1), gt(:, 2));
    plot(store(1,1), store(1,2), 'o');
    legend({'estimate', 'ground truth', 'starting point'}, 'FontSize', 14);
    xlabel('x (m)', 'FontSize', 14);
    ylabel('y (m)', 'FontSize', 14);
    xlim([-inf inf])
    ylim([-inf inf])
    hold off;
% v
end

function show_plots(gt, z, xk, zModel)
    
    figure(1);
    hold on;
    title('Course Correction EKF Trajectory Estimation');
    plot(xk(:,1), xk(:, 2));
    plot(gt(:,1), gt(:, 2));
    legend('estimate', 'ground truth');
    xlabel('x (m)');
    ylabel('y (m)');
    xlim([-inf inf])
    ylim([-inf inf])
    hold off;

%     figure(2);
%     hold on;
%     plot(z(:,1), z(:, 2));
%     plot(zModel(:,1), zModel(:, 2));    
%     legend('Measurement', 'Measurement Model');
%     xlim([-inf inf])
%     ylim([-inf inf])
%     hold off;

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