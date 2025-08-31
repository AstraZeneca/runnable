// Set up SVG dimensions
const width = 900;
const height = 600;

const svg = d3.select("#workflow-graph")
    .attr("width", width)
    .attr("height", height);

// Set up force simulation
const nodeSpacing = width / (graphData.nodes.length + 1);
const simulation = d3.forceSimulation(graphData.nodes)
    .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(100))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2))
    // Add horizontal force based on node index
    .force("x", d3.forceX((d, i) => nodeSpacing * (i + 1)).strength(1))
    .force("y", d3.forceY(height / 2).strength(0.1));

// Define arrow markers for directed links
svg.append("defs").selectAll("marker")
    .data(["end"]) // Only one type of marker for now
    .enter().append("marker")
    .attr("id", String)
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 36) // Offset marker so it doesn't overlap node (node image is 32px, so 36 is just outside)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", "#999");

// Create links
const link = svg.append("g")
    .attr("class", "links")
    .selectAll("line")
    .data(graphData.links)
    .enter().append("line")
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.6)
    .attr("marker-end", "url(#end)");

// --- Create Nodes as Images ---
const node = svg.append("g")
    .attr("class", "nodes")
    .selectAll("g")
    .data(graphData.nodes)
    .enter().append("g")
    .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

// Append IMAGES for nodes
node.append("image")
    .attr("xlink:href", d => {
        // Determine the image path based on node label or task_type
        if (d.task_type === "python") return "/static/images/python.png";
        if (d.task_type === "notebook") return "/static/images/notebook.png";
        if (d.task_type === "shell") return "/static/images/shell.png";
        if (d.label === "parallel") return "/static/images/parallel.png"; // Assuming you have a parallel icon
        if (d.label === "success") return "/static/images/success.png";   // Assuming you have a success icon
        return "/static/images/default.png"; // Fallback image
    })
    .attr("x", -16) // Adjust x, y, width, height to center the image
    .attr("y", -16) // This assumes image size is 32x32px
    .attr("width", 32)
    .attr("height", 32)
    .attr("class", d => `node-image node-type-${d.label} ${d.task_type ? 'task-subtype-' + d.task_type : ''}`);

// Add a text label *below* the image for better readability
node.append("text")
    .attr("dy", "2.5em") // Position below the image (assuming 32px image + padding)
    .attr("text-anchor", "middle") // Center the text horizontally
    .text(d => d.id)
    .attr("font-size", "10px")
    .attr("fill", "#555")
    .attr("pointer-events", "none"); // Allow interaction to pass through to the image


// Tooltip functionality
const tooltip = d3.select("#tooltip");

node.on("mouseover", function(event, d) {
    tooltip.html(getNodeTooltipContent(d))
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 20) + "px")
        .style("opacity", 1);
})
.on("mouseout", function() {
    tooltip.style("opacity", 0);
});

// Update positions on each simulation tick
simulation.on("tick", () => {

    // Calculate link endpoints so arrows stop at edge of node image (radius = 16)
    link
        .attr("x1", d => {
            const dx = d.target.x - d.source.x;
            const dy = d.target.y - d.source.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const r = 16; // node image radius
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

    node
        .attr("transform", d => `translate(${d.x},${d.y})`);
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
    // You can uncomment these lines if you want nodes to stay fixed after drag,
    // otherwise they will slowly move back to a stable position.
    // d.fx = null;
    // d.fy = null;
}

function getNodeTooltipContent(d) {
    let content = `<strong>ID:</strong> ${d.id}<br/>`;
    content += `<strong>Type:</strong> ${d.label}<br/>`;
    if (d.task_type) {
        content += `<strong>Task Subtype:</strong> ${d.task_type}<br/>`;
        content += `<strong>Metadata:</strong><br/>`;
        for (const key in d.metadata) {
            content += `&nbsp;&nbsp;&nbsp;${key}: ${JSON.stringify(d.metadata[key])}<br/>`;
        }
    }
    return content;
}
