// D3.js visualization script
const svgElement = document.getElementById("workflow-graph");
const width = svgElement.clientWidth;
const height = svgElement.clientHeight;
const svg = d3.select(svgElement);

// Define arrow markers for directed links
svg.append("defs").selectAll("marker")
    .data(["end"])
    .enter().append("marker")
    .attr("id", "end")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 20)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", "#6b7280");

// Modal functionality
function createModal() {
    const modalHtml = `
        <div id="metadataModal" class="hidden fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
            <div class="relative top-20 mx-auto p-5 border w-11/12 md:w-3/4 lg:w-1/2 shadow-lg rounded-md bg-white">
                <div class="flex justify-between items-center">
                    <h2 class="text-xl font-bold text-gray-700">Node Metadata</h2>
                    <button class="close-modal text-gray-400 bg-transparent hover:bg-gray-200 hover:text-gray-900 rounded-lg text-sm w-8 h-8 flex justify-center items-center">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>
                <div class="modal-body mt-4 text-gray-600"></div>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', modalHtml);
}

function showModal(content) {
    const modal = document.getElementById('metadataModal');
    const modalBody = modal.querySelector('.modal-body');
    modalBody.innerHTML = content;
    modal.classList.remove('hidden');
}

function setupModalEvents() {
    const modal = document.getElementById('metadataModal');
    const closeBtn = modal.querySelector('.close-modal');

    closeBtn.onclick = () => modal.classList.add('hidden');
    window.onclick = (event) => {
        if (event.target === modal) {
            modal.classList.add('hidden');
        }
    };
}

// Create modal on page load
createModal();
setupModalEvents();

// Tooltip functionality
const tooltip = d3.select("#tooltip");

function getNodeTooltipContent(d) {
    let content = `<div class="space-y-2">
        <div><span class="font-semibold">ID:</span> ${d.id}</div>
        <div><span class="font-semibold">Type:</span> ${d.label}</div>`;
    if (d.task_type) {
        content += `
        <div><span class="font-semibold">Task Subtype:</span> ${d.task_type}</div>
        <div class="mt-4">
            <div class="font-semibold mb-2">Metadata:</div>
            <div class="pl-4 space-y-1">`;
        for (const key in d.metadata) {
            content += `<div><span class="text-gray-700">${key}:</span> ${JSON.stringify(d.metadata[key])}</div>`;
        }
        content += `</div></div>`;
    }
    content += `</div>`;
    return content;
}

function renderGraph(graphData) {
    // A simple topological sort to order nodes for a left-to-right layout
    const nodeMap = new Map(graphData.nodes.map(d => [d.id, { ...d, dependencies: 0, dependents: 0 }]));
    graphData.links.forEach(link => {
        nodeMap.get(link.target).dependencies++;
        nodeMap.get(link.source).dependents++;
    });

    // Identify start nodes (no incoming links)
    let startNodes = Array.from(nodeMap.values()).filter(d => d.dependencies === 0);
    let queue = [...startNodes];
    const sortedNodes = [];
    const visited = new Set();
    const levels = new Map();
    let currentLevel = 0;

    while (queue.length > 0) {
        const levelSize = queue.length;
        for (let i = 0; i < levelSize; i++) {
            const node = queue.shift();
            if (!visited.has(node.id)) {
                visited.add(node.id);
                sortedNodes.push(node);
                levels.set(node.id, currentLevel);

                graphData.links.filter(link => link.source === node.id).forEach(link => {
                    queue.push(nodeMap.get(link.target));
                });
            }
        }
        currentLevel++;
    }

    // Map the sorted nodes back to the original array to maintain order
    const sortedGraphDataNodes = graphData.nodes.sort((a, b) => levels.get(a.id) - levels.get(b.id));

    // Set up force simulation with horizontal positioning
    const simulation = d3.forceSimulation(sortedGraphDataNodes)
        .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("x", d3.forceX(d => {
            const level = levels.get(d.id);
            const totalLevels = Math.max(...Array.from(levels.values())) + 1;
            return (width / totalLevels) * (level + 0.5);
        }).strength(0.5))
        .force("y", d3.forceY(height / 2).strength(0.1));

    // Create links
    const link = svg.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(graphData.links)
        .enter().append("line")
        .attr("stroke", "#9ca3af")
        .attr("stroke-width", 2)
        .attr("stroke-opacity", 0.8)
        .attr("marker-end", "url(#end)");

    // Create Nodes as Images with text labels
    const node = svg.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(sortedGraphDataNodes)
        .enter().append("g")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended))
        .on("click", (event, d) => {
            // Prevent modal from opening if we're dragging
            if (event.defaultPrevented) return;
            // Show modal with detailed metadata
            const content = getNodeTooltipContent(d);
            showModal(content);
        });

    // Append IMAGES for nodes
    node.append("image")
        .attr("xlink:href", d => {
            // Use relative paths to your local images
            if (d.task_type === "python") return "static/images/python.png";
            if (d.task_type === "notebook") return "static/images/notebook.png";
            if (d.task_type === "shell") return "static/images/shell.png";
            if (d.label === "parallel") return "static/images/parallel.png";
            if (d.label === "success") return "static/images/success.png";
            // Provide a default image for unknown types
            return "static/images/default.png";
        })
        .attr("width", 32)
        .attr("height", 32)
        .attr("class", d => `node-image ${d.label} ${d.task_type ? 'task-' + d.task_type : ''}`);

    // Add a text label below the image
    node.append("text")
        .attr("dy", "2.5em")
        .attr("text-anchor", "middle")
        .text(d => d.id)
        .attr("font-size", "10px")
        .attr("fill", "#4b5563")
        .attr("pointer-events", "none");

    // Tooltip functionality
    node.on("mouseover", function(event, d) {
        tooltip.html(getNodeTooltipContent(d))
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 20) + "px")
            .style("display", "block")
            .style("opacity", 1);
    })
    .on("mouseout", function() {
        tooltip.style("opacity", 0)
            .transition()
            .delay(200)
            .style("display", "none");
    });

    // Update positions on each simulation tick
    simulation.on("tick", () => {
        link
            .attr("x1", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const r = 16;
                return d.source.x + (dx * r) / dist;
            })
            .attr("y1", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const r = 16;
                return d.source.y + (dy * r) / dist;
            })
            .attr("x2", d => {
                const dx = d.source.x - d.target.x;
                const dy = d.source.y - d.target.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const r = 16;
                return d.target.x + (dx * r) / dist;
            })
            .attr("y2", d => {
                const dx = d.source.x - d.target.x;
                const dy = d.source.y - d.target.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const r = 16;
                return d.target.y + (dy * r) / dist;
            });

        node.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    // Drag functions
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
    }
}

// Fetch graph data from the API and render the graph
async function fetchAndRenderGraph() {
    try {
        const response = await fetch('/graph-data');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const graphData = await response.json();
        renderGraph(graphData);
    } catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
    }
}

fetchAndRenderGraph();


// Handle window resize to make the SVG responsive
window.addEventListener('resize', () => {
    const newWidth = svgElement.clientWidth;
    const newHeight = svgElement.clientHeight;
    svg.attr("width", newWidth).attr("height", newHeight);
    simulation.force("center", d3.forceCenter(newWidth / 2, newHeight / 2));
    simulation.alpha(0.3).restart();
});
