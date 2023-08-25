x = z(:,1);
y = z(:,2);
theta = z(:,3);

interpHeading = [0];

for i=2:length(x)
    interpHeading(i) = atan2(y(i) - y(i-1), x(i) - x(i-1));
end

test = [theta, interpHeading'];