const canvas = document.getElementById('pendulumCanvas');
const ctx = canvas.getContext('2d');
const CANVAS_WIDTH = canvas.width;
const CANVAS_HEIGHT = canvas.height;
const CENTER_X = CANVAS_WIDTH / 2;
const CENTER_Y = CANVAS_HEIGHT / 4; // Pivot point for pendulums

let animationFrameId;
let currentFrame = 0;
let simulationData = {};
let currentPendulumType = 'simple';

// --- Utility Functions ---
function degToRad(degrees) {
    return degrees * Math.PI / 180;
}

function radToDeg(radians) {
    return radians * 180 / Math.PI;
}

// --- RK4 Solver (4th Order Runge-Kutta) ---
// A simplified RK4 for fixed step size, suitable for this simulation.
// For adaptive step size like scipy's solve_ivp, a more complex implementation is needed.
function solveRK4(ode_func, y0, t_span, args, t_eval) {
    const t_start = t_span[0];
    const t_end = t_span[1];
    const h = t_eval[1] - t_eval[0]; // Fixed step size

    let y = [...y0]; // Current state
    const results = [y0.slice()]; // Store initial state

    for (let i = 0; i < t_eval.length - 1; i++) {
        const t = t_eval[i];

        const k1 = ode_func(t, y, ...args).map(val => val * h);
        const y1 = y.map((val, idx) => val + k1[idx] / 2);

        const k2 = ode_func(t + h / 2, y1, ...args).map(val => val * h);
        const y2 = y.map((val, idx) => val + k2[idx] / 2);

        const k3 = ode_func(t + h / 2, y2, ...args).map(val => val * h);
        const y3 = y.map((val, idx) => val + k3[idx]);

        const k4 = ode_func(t + h, y3, ...args).map(val => val * h);

        y = y.map((val, idx) => val + (k1[idx] + 2 * k2[idx] + 2 * k3[idx] + k4[idx]) / 6);
        results.push(y.slice());
    }
    return { y: results };
}

// --- ODE Definitions ---

// 1. Simple Pendulum
function simple_pendulum_ode(t, y, L, g, gamma) {
    const [theta, omega] = y;
    const dtheta_dt = omega;
    const domega_dt = - (g / L) * Math.sin(theta) - gamma * omega;
    return [dtheta_dt, domega_dt];
}

function simulate_simple_pendulum(L, g, initial_angle_deg, initial_angular_velocity, gamma, duration, fps) {
    const initial_angle_rad = degToRad(initial_angle_deg);
    const y0 = [initial_angle_rad, initial_angular_velocity];
    const t_eval = Array.from({ length: Math.floor(duration * fps) + 1 }, (_, i) => i / fps);

    const sol = solveRK4(
        simple_pendulum_ode,
        y0,
        [0, duration],
        [L, g, gamma],
        t_eval
    );

    const theta = sol.y.map(state => state[0]);
    const x = theta.map(t => L * Math.sin(t));
    const y = theta.map(t => -L * Math.cos(t));

    const period_small_angle = 2 * Math.PI * Math.sqrt(L / g);

    return { t_eval, x, y, theta, period_small_angle };
}

// 2. Compound Pendulum
function compound_pendulum_ode(t, y, M, d, I, g, gamma) {
    const [theta, omega] = y;
    const dtheta_dt = omega;
    const domega_dt = - (M * g * d / I) * Math.sin(theta) - gamma * omega;
    return [dtheta_dt, domega_dt];
}

function simulate_compound_pendulum(M, d, I, g, initial_angle_deg, initial_angular_velocity, gamma, duration, fps) {
    const initial_angle_rad = degToRad(initial_angle_deg);
    const y0 = [initial_angle_rad, initial_angular_velocity];
    const t_eval = Array.from({ length: Math.floor(duration * fps) + 1 }, (_, i) => i / fps);

    const sol = solveRK4(
        compound_pendulum_ode,
        y0,
        [0, duration],
        [M, d, I, g, gamma],
        t_eval
    );

    const theta = sol.y.map(state => state[0]);
    const L_eq = (M * d !== 0) ? I / (M * d) : 1.0; // Equivalent length for visualization
    const x = theta.map(t => L_eq * Math.sin(t));
    const y = theta.map(t => -L_eq * Math.cos(t));

    const period_compound = (M * g * d !== 0) ? 2 * Math.PI * Math.sqrt(I / (M * g * d)) : Infinity;

    return { t_eval, x, y, theta, L_eq, period_compound };
}

// 3. Conical Pendulum
function conical_pendulum_ode(t, y, L, g, initial_cone_angle_rad) {
    const [phi, dphi_dt] = y;
    let omega_conical;
    if (initial_cone_angle_rad === 0) {
        omega_conical = 0;
    } else {
        omega_conical = Math.sqrt(g / (L * Math.cos(initial_cone_angle_rad)));
    }
    return [omega_conical, 0]; // dphi/dt is constant, d(dphi/dt)/dt is 0
}

function simulate_conical_pendulum(L, g, initial_cone_angle_deg, duration, fps) {
    const initial_cone_angle_rad = degToRad(initial_cone_angle_deg);

    let omega_conical;
    if (initial_cone_angle_rad === 0) {
        omega_conical = 0;
    } else {
        omega_conical = Math.sqrt(g / (L * Math.cos(initial_cone_angle_rad)));
    }

    const y0 = [0, omega_conical]; // Start at phi=0 with calculated angular velocity
    const t_eval = Array.from({ length: Math.floor(duration * fps) + 1 }, (_, i) => i / fps);

    const sol = solveRK4(
        conical_pendulum_ode,
        y0,
        [0, duration],
        [L, g, initial_cone_angle_rad],
        t_eval
    );

    const phi = sol.y.map(state => state[0]);

    // Convert to Cartesian coordinates (3D) for visualization
    const x = phi.map(p => L * Math.sin(initial_cone_angle_rad) * Math.cos(p));
    const y = phi.map(p => L * Math.sin(initial_cone_angle_rad) * Math.sin(p));
    const z = phi.map(p => -L * Math.cos(initial_cone_angle_rad)); // Z is negative downwards

    const period_conical = (initial_cone_angle_rad < Math.PI / 2) ? 2 * Math.PI * Math.sqrt(L * Math.cos(initial_cone_angle_rad) / g) : Infinity;

    return { t_eval, x, y, z, initial_cone_angle_rad, period_conical };
}

// 4. Double Pendulum
function double_pendulum_ode(t, y, L1, L2, M1, M2, g) {
    const [theta1, omega1, theta2, omega2] = y;

    const delta = theta1 - theta2;

    const dtheta1_dt = omega1;
    const dtheta2_dt = omega2;

    const den1 = L1 * (2 * M1 + M2 - M2 * Math.cos(2 * delta));
    const domega1_dt = (-g * (2 * M1 + M2) * Math.sin(theta1) - M2 * g * Math.sin(theta1 - 2 * theta2) - 2 * Math.sin(delta) * M2 * (omega2**2 * L2 + omega1**2 * L1 * Math.cos(delta))) / den1;

    const den2 = L2 * (2 * M1 + M2 - M2 * Math.cos(2 * delta));
    const domega2_dt = (2 * Math.sin(delta) * (omega1**2 * L1 * (M1 + M2) + g * (M1 + M2) * Math.cos(theta1) + omega2**2 * L2 * M2 * Math.cos(delta))) / den2;

    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt];
}

function simulate_double_pendulum(L1, L2, M1, M2, initial_angle1_deg, initial_angle2_deg, duration, fps, g) {
    const initial_angle1_rad = degToRad(initial_angle1_deg);
    const initial_angle2_rad = degToRad(initial_angle2_deg);

    const y0 = [initial_angle1_rad, 0, initial_angle2_rad, 0]; // Start with zero angular velocities
    const t_eval = Array.from({ length: Math.floor(duration * fps) + 1 }, (_, i) => i / fps);

    const sol = solveRK4(
        double_pendulum_ode,
        y0,
        [0, duration],
        [L1, L2, M1, M2, g],
        t_eval
    );

    const theta1 = sol.y.map(state => state[0]);
    const theta2 = sol.y.map(state => state[2]);

    const x1 = theta1.map(t => L1 * Math.sin(t));
    const y1 = theta1.map(t => -L1 * Math.cos(t));

    const x2 = theta2.map((t2, i) => x1[i] + L2 * Math.sin(t2));
    const y2 = theta2.map((t2, i) => y1[i] - L2 * Math.cos(t2));

    return { t_eval, x1, y1, x2, y2 };
}

// --- Drawing Functions ---
const SCALE_FACTOR = 100; // Pixels per meter

function clearCanvas() {
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    ctx.fillStyle = '#fdfdfd';
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
}

function drawSimplePendulum(data, frame) {
    clearCanvas();
    const x = data.x[frame] * SCALE_FACTOR + CENTER_X;
    const y = data.y[frame] * SCALE_FACTOR + CENTER_Y;

    // Pivot
    ctx.beginPath();
    ctx.arc(CENTER_X, CENTER_Y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = '#333';
    ctx.fill();

    // Rod
    ctx.beginPath();
    ctx.moveTo(CENTER_X, CENTER_Y);
    ctx.lineTo(x, y);
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Bob
    ctx.beginPath();
    ctx.arc(x, y, 15, 0, 2 * Math.PI);
    ctx.fillStyle = '#007bff';
    ctx.fill();
    ctx.strokeStyle = '#0056b3';
    ctx.lineWidth = 1;
    ctx.stroke();
}

function drawCompoundPendulum(data, frame) {
    clearCanvas();
    const x = data.x[frame] * SCALE_FACTOR + CENTER_X;
    const y = data.y[frame] * SCALE_FACTOR + CENTER_Y;

    // Pivot
    ctx.beginPath();
    ctx.arc(CENTER_X, CENTER_Y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = '#333';
    ctx.fill();

    // Rod (represented by equivalent length)
    ctx.beginPath();
    ctx.moveTo(CENTER_X, CENTER_Y);
    ctx.lineTo(x, y);
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 4; // Thicker to represent a body
    ctx.stroke();

    // Center of Mass (bob)
    ctx.beginPath();
    ctx.arc(x, y, 20, 0, 2 * Math.PI); // Larger bob for compound
    ctx.fillStyle = '#28a745';
    ctx.fill();
    ctx.strokeStyle = '#1e7e34';
    ctx.lineWidth = 1;
    ctx.stroke();
}

function drawConicalPendulum(data, frame) {
    clearCanvas();
    const x = data.x[frame] * SCALE_FACTOR + CENTER_X;
    const y = data.y[frame] * SCALE_FACTOR + CENTER_Y;
    const z = data.z[frame] * SCALE_FACTOR; // Z-coordinate for perspective

    // Simple perspective effect: scale and offset based on Z
    const perspectiveFactor = 1 + z / (SCALE_FACTOR * 5); // Adjust 5 for desired perspective
    const bobRadius = 15 * perspectiveFactor;
    const bobYOffset = z * 0.5; // Adjust for vertical perspective shift

    // Pivot
    ctx.beginPath();
    ctx.arc(CENTER_X, CENTER_Y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = '#333';
    ctx.fill();

    // Rod
    ctx.beginPath();
    ctx.moveTo(CENTER_X, CENTER_Y);
    ctx.lineTo(x, y + bobYOffset); // Apply Y offset for perspective
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Bob
    ctx.beginPath();
    ctx.arc(x, y + bobYOffset, bobRadius, 0, 2 * Math.PI);
    ctx.fillStyle = '#ffc107';
    ctx.fill();
    ctx.strokeStyle = '#e0a800';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Draw the circular path on the "floor" (below the pivot)
    ctx.beginPath();
    ctx.arc(CENTER_X, CENTER_Y + data.L * Math.cos(data.initial_cone_angle_rad) * SCALE_FACTOR,
            data.L * Math.sin(data.initial_cone_angle_rad) * SCALE_FACTOR, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
    ctx.lineWidth = 1;
    ctx.stroke();
}

function drawDoublePendulum(data, frame) {
    clearCanvas();
    const x1 = data.x1[frame] * SCALE_FACTOR + CENTER_X;
    const y1 = data.y1[frame] * SCALE_FACTOR + CENTER_Y;
    const x2 = data.x2[frame] * SCALE_FACTOR + CENTER_X;
    const y2 = data.y2[frame] * SCALE_FACTOR + CENTER_Y;

    // Pivot
    ctx.beginPath();
    ctx.arc(CENTER_X, CENTER_Y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = '#333';
    ctx.fill();

    // Rod 1
    ctx.beginPath();
    ctx.moveTo(CENTER_X, CENTER_Y);
    ctx.lineTo(x1, y1);
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Bob 1
    ctx.beginPath();
    ctx.arc(x1, y1, 15, 0, 2 * Math.PI);
    ctx.fillStyle = '#17a2b8';
    ctx.fill();
    ctx.strokeStyle = '#138496';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Rod 2
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Bob 2
    ctx.beginPath();
    ctx.arc(x2, y2, 15, 0, 2 * Math.PI);
    ctx.fillStyle = '#dc3545';
    ctx.fill();
    ctx.strokeStyle = '#c82333';
    ctx.lineWidth = 1;
    ctx.stroke();
}

// --- Animation Loop ---
function animate() {
    if (!simulationData.t_eval || simulationData.t_eval.length === 0) {
        console.warn("No simulation data to animate.");
        return;
    }

    const totalFrames = simulationData.t_eval.length;
    const drawFunction = getDrawFunction(currentPendulumType);

    drawFunction(simulationData, currentFrame);

    currentFrame = (currentFrame + 1) % totalFrames;
    animationFrameId = requestAnimationFrame(animate);
}

function stopAnimation() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

// --- Control and Simulation Logic ---
const pendulumTypeSelect = document.getElementById('pendulumType');
const commonParamsDiv = document.getElementById('common-params');
const simpleParamsDiv = document.getElementById('simple-pendulum-params');
const compoundParamsDiv = document.getElementById('compound-pendulum-params');
const conicalParamsDiv = document.getElementById('conical-pendulum-params');
const doubleParamsDiv = document.getElementById('double-pendulum-params');
const startButton = document.getElementById('startSimulation');
const stopButton = document.getElementById('stopSimulation');
const periodInfo = document.getElementById('period-info');
const conicalPeriodInfo = document.getElementById('conical-period-info');

const allParamDivs = {
    'simple': simpleParamsDiv,
    'compound': compoundParamsDiv,
    'conical': conicalParamsDiv,
    'double': doubleParamsDiv
};

function updateControlVisibility() {
    for (const type in allParamDivs) {
        if (type === currentPendulumType) {
            allParamDivs[type].classList.remove('hidden');
        } else {
            allParamDivs[type].classList.add('hidden');
        }
    }
    // Specific result info visibility
    if (currentPendulumType === 'conical') {
        periodInfo.classList.add('hidden');
        conicalPeriodInfo.classList.remove('hidden');
    } else {
        periodInfo.classList.remove('hidden');
        conicalPeriodInfo.classList.add('hidden');
    }
}

function getParams() {
    const params = {
        g: parseFloat(document.getElementById('g').value),
        duration: parseFloat(document.getElementById('duration').value),
        fps: parseInt(document.getElementById('fps').value)
    };

    switch (currentPendulumType) {
        case 'simple':
            params.L = parseFloat(document.getElementById('simple-L').value);
            params.initial_angle_deg = parseFloat(document.getElementById('simple-initial_angle_deg').value);
            params.initial_angular_velocity = parseFloat(document.getElementById('simple-initial_angular_velocity').value);
            params.gamma = parseFloat(document.getElementById('simple-gamma').value);
            break;
        case 'compound':
            params.M = parseFloat(document.getElementById('compound-M').value);
            params.d = parseFloat(document.getElementById('compound-d').value);
            params.I = parseFloat(document.getElementById('compound-I').value);
            params.initial_angle_deg = parseFloat(document.getElementById('compound-initial_angle_deg').value);
            params.initial_angular_velocity = parseFloat(document.getElementById('compound-initial_angular_velocity').value);
            params.gamma = parseFloat(document.getElementById('compound-gamma').value);
            break;
        case 'conical':
            params.L = parseFloat(document.getElementById('conical-L').value);
            params.initial_cone_angle_deg = parseFloat(document.getElementById('conical-initial_cone_angle_deg').value);
            break;
        case 'double':
            params.L1 = parseFloat(document.getElementById('double-L1').value);
            params.M1 = parseFloat(document.getElementById('double-M1').value);
            params.initial_angle1_deg = parseFloat(document.getElementById('double-initial_angle1_deg').value);
            params.L2 = parseFloat(document.getElementById('double-L2').value);
            params.M2 = parseFloat(document.getElementById('double-M2').value);
            params.initial_angle2_deg = parseFloat(document.getElementById('double-initial_angle2_deg').value);
            break;
    }
    return params;
}

function getSimulateFunction(type) {
    switch (type) {
        case 'simple': return simulate_simple_pendulum;
        case 'compound': return simulate_compound_pendulum;
        case 'conical': return simulate_conical_pendulum;
        case 'double': return simulate_double_pendulum;
        default: return null;
    }
}

function getDrawFunction(type) {
    switch (type) {
        case 'simple': return drawSimplePendulum;
        case 'compound': return drawCompoundPendulum;
        case 'conical': return drawConicalPendulum;
        case 'double': return drawDoublePendulum;
        default: return null;
    }
}

function runSimulation() {
    stopAnimation();
    currentFrame = 0;
    const params = getParams();
    const simulateFunc = getSimulateFunction(currentPendulumType);

    if (!simulateFunc) {
        console.error("Invalid pendulum type selected.");
        return;
    }

    switch (currentPendulumType) {
        case 'simple':
            simulationData = simulateFunc(params.L, params.g, params.initial_angle_deg, params.initial_angular_velocity, params.gamma, params.duration, params.fps);
            periodInfo.textContent = `Period (small angle approx): ${simulationData.period_small_angle.toFixed(3)} s`;
            break;
        case 'compound':
            simulationData = simulateFunc(params.M, params.d, params.I, params.g, params.initial_angle_deg, params.initial_angular_velocity, params.gamma, params.duration, params.fps);
            periodInfo.textContent = `Period (small angle approx): ${simulationData.period_compound.toFixed(3)} s`;
            break;
        case 'conical':
            simulationData = simulateFunc(params.L, params.g, params.initial_cone_angle_deg, params.duration, params.fps);
            conicalPeriodInfo.textContent = `Conical Period: ${simulationData.period_conical.toFixed(3)} s`;
            // Store L and initial_cone_angle_rad for drawing
            simulationData.L = params.L;
            simulationData.initial_cone_angle_rad = degToRad(params.initial_cone_angle_deg);
            break;
        case 'double':
            simulationData = simulateFunc(params.L1, params.L2, params.M1, params.M2, params.initial_angle1_deg, params.initial_angle2_deg, params.duration, params.fps, params.g);
            periodInfo.textContent = `Double pendulums exhibit chaotic motion, no simple period.`;
            break;
    }
    animate();
}

// --- Event Listeners ---
pendulumTypeSelect.addEventListener('change', (event) => {
    currentPendulumType = event.target.value;
    updateControlVisibility();
    runSimulation(); // Re-run simulation with new type
});

// Update value displays for sliders
document.querySelectorAll('input[type="range"]').forEach(slider => {
    const valueSpan = document.getElementById(slider.id + '-value');
    if (valueSpan) {
        slider.addEventListener('input', () => {
            let unit = '';
            if (slider.id.includes('angle')) unit = '°';
            else if (slider.id.includes('L')) unit = ' m';
            else if (slider.id.includes('M')) unit = ' kg';
            else if (slider.id.includes('d')) unit = ' m';
            else if (slider.id.includes('I')) unit = ' kg·m²';
            else if (slider.id.includes('gamma')) unit = ' s⁻¹';
            else if (slider.id === 'g') unit = ' m/s²';
            else if (slider.id === 'duration') unit = ' s';
            else if (slider.id === 'fps') unit = '';

            valueSpan.textContent = `${slider.value}${unit}`;
        });
        slider.addEventListener('change', runSimulation); // Re-run on change
    }
});

startButton.addEventListener('click', runSimulation);
stopButton.addEventListener('click', stopAnimation);

// Initial setup
updateControlVisibility();
// Initialize slider value displays
document.querySelectorAll('input[type="range"]').forEach(slider => {
    const valueSpan = document.getElementById(slider.id + '-value');
    if (valueSpan) {
        let unit = '';
        if (slider.id.includes('angle')) unit = '°';
        else if (slider.id.includes('L')) unit = ' m';
        else if (slider.id.includes('M')) unit = ' kg';
        else if (slider.id.includes('d')) unit = ' m';
        else if (slider.id.includes('I')) unit = ' kg·m²';
        else if (slider.id.includes('gamma')) unit = ' s⁻¹';
        else if (slider.id === 'g') unit = ' m/s²';
        else if (slider.id === 'duration') unit = ' s';
        else if (slider.id === 'fps') unit = '';
        valueSpan.textContent = `${slider.value}${unit}`;
    }
});
runSimulation(); // Run initial simulation
