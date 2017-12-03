import { Component, OnInit } from '@angular/core';
import * as d3 from 'd3';
import * as d3scale from 'd3-scale';
import { Axis, axisBottom, axisRight } from 'd3-axis';
import { select, Selection } from 'd3-selection';
//import * as _ from 'seedrandom';
//import seedrandom = require('seedrandom');

import * as nn from './nn';
import {HeatMap, reduceMatrix} from './heatmap';
import {
  State,
  datasets,
  regDatasets,
  activations,
  problems,
  regularizations,
  getKeyFromValue,
  Problem
} from './state';
import {Example2D, shuffle} from './dataset';
import {AppendingLineChart} from './linechart';


declare let ga: any;

// More scrolling
d3.select('.more button').on('click', function() {
  const position = 800;
  d3.transition()
    .duration(1000)
    .tween('scroll', scrollTween(position));
});

function scrollTween(offset) {
  return function() {
    const i = d3.interpolateNumber(window.pageYOffset ||
        document.documentElement.scrollTop, offset);
    return function(t) { scrollTo(0, i(t)); };
  };
}


enum HoverType {
  BIAS, WEIGHT
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

@Component({
  selector: 'app-playground',
  templateUrl: './playground.component.html',
  styleUrls: ['./playground.component.css']
})
export class PlaygroundComponent implements OnInit {

  private static RECT_SIZE = 30;
  private static  BIAS_SIZE = 5;
  private static  NUM_SAMPLES_CLASSIFY = 500;
  private static  NUM_SAMPLES_REGRESS = 1200;
  private static  DENSITY = 100;
  
  private static  INPUTS: {[name: string]: InputFeature} = {
    'x': {f: (x, y) => x, label: 'X_1'},
    'y': {f: (x, y) => y, label: 'X_2'},
    'xSquared': {f: (x, y) => x * x, label: 'X_1^2'},
    'ySquared': {f: (x, y) => y * y,  label: 'X_2^2'},
    'xTimesY': {f: (x, y) => x * y, label: 'X_1X_2'},
    'sinX': {f: (x, y) => Math.sin(x), label: 'sin(X_1)'},
    'sinY': {f: (x, y) => Math.sin(y), label: 'sin(X_2)'},
  };
  
  private static HIDABLE_CONTROLS = [
    ['Show test data', 'showTestData'],
    ['Discretize output', 'discretize'],
    ['Play button', 'playButton'],
    ['Step button', 'stepButton'],
    ['Reset button', 'resetButton'],
    ['Learning rate', 'learningRate'],
    ['Activation', 'activation'],
    ['Regularization', 'regularization'],
    ['Regularization rate', 'regularizationRate'],
    ['Problem type', 'problem'],
    ['Which dataset', 'dataset'],
    ['Ratio train data', 'percTrainData'],
    ['Noise level', 'noise'],
    ['Batch size', 'batchSize'],
    ['# of hidden layers', 'numHiddenLayers'],
  ];
  private state;
  private firstInteraction = true;
  private parametersChanged = false;
  private colorScale;
  private iter = 0;
  private linkWidthScale;
  private selectedNodeId: string;
  private heatMap;
  private trainData: Example2D[];
  private testData: Example2D[];
  private network: nn.Node[][];
  private lossTrain;
  private lossTest;
  private lineChart;
  private boundary: {[id: string]: number[][]};
  private xDomain: [number, number];
  private timerIndex = 0;
  private isPlaying = false;

  constructor() { }

  ngOnInit() {
    this.state = State.deserializeState();
    let state = this.state;
    // Filter out inputs that are hidden.
    state.getHiddenProps().forEach(prop => {
      if (prop in PlaygroundComponent.INPUTS) {
        delete PlaygroundComponent.INPUTS[prop];
      }
    });

    this.boundary = {};
    this.selectedNodeId = null;
    // Plot the heatmap.
    this.xDomain = [-6, 6];
    this.heatMap =
        new HeatMap(300, PlaygroundComponent.DENSITY, this.xDomain, this.xDomain, d3.select('#heatmap'),
            {showAxes: true});
    this.linkWidthScale = d3.scaleLinear()
      .domain([0, 5])
      .range([1, 10])
      .clamp(true);
    this.colorScale = d3.scaleLinear<string>()
                        .domain([-1, 0, 1])
                        .range(['#f59322', '#e8eaeb', '#0877bd'])
                        .clamp(true);
    this.trainData = [];
    this.testData = [];
    this.network = null;
    this.lossTrain = 0;
    this.lossTest = 0;
    this.lineChart = new AppendingLineChart(d3.select('#linechart'),
        ['#777', 'black']);
    let that = this;
    d3.select('#play-pause-button').on('click', function () {
      // Change the button's content.
      that.userHasInteracted();
      that.playOrPause();
    });

    d3.select('#next-step-button').on('click', () => {
      that.pause();
      that.userHasInteracted();
      if (this.iter === 0) {
        that.simulationStarted();
      }
      that.oneStep();
    });


    d3.select('#data-regen-button').on('click', () => {
      that.generateData();
      that.parametersChanged = true;
    });

    let dataThumbnails = d3.selectAll('canvas[data-dataset]');
    dataThumbnails.on('click', function(){
      let newDataset = datasets[state.dataset.dataset];
      if (newDataset === state.dataset) {
        return; // No-op.
      }
      state.dataset =  newDataset;
      dataThumbnails.classed('selected', false);
      d3.select(this).classed('selected', true);
      that.generateData();
      that.parametersChanged = true;
      that.reset();
    });

    let datasetKey = getKeyFromValue(datasets, state.dataset);
    // Select the dataset according to the current state.
    d3.select(`canvas[data-dataset=${datasetKey}]`)
      .classed('selected', true);

    let regDataThumbnails = d3.selectAll('canvas[data-regDataset]');
    regDataThumbnails.on('click', () => {
      let newDataset = regDatasets[state.dataset.regdataset];
      if (newDataset === state.regDataset) {
        return; // No-op.
      }
      state.regDataset =  newDataset;
      regDataThumbnails.classed('selected', false);
      //d3.select(this).classed('selected', true);
      that.generateData();
      that.parametersChanged = true;
      that.reset();
    });

    let regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
    // Select the dataset according to the current state.
    d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
      .classed('selected', true);

    d3.select('#add-layers').on('click', () => {
      if (state.numHiddenLayers >= 6) {
        return;
      }
      state.networkShape[state.numHiddenLayers] = 2;
      state.numHiddenLayers++;
      that.parametersChanged = true;
      that.reset();
    });

    d3.select('#remove-layers').on('click', () => {
      if (state.numHiddenLayers <= 0) {
        return;
      }
      state.numHiddenLayers--;
      state.networkShape.splice(state.numHiddenLayers);
      that.parametersChanged = true;
      that.reset();
    });

    let showTestData = d3.select('#show-test-data').on('change', function(){
      state.showTestData = d3.select(this).attr('checked');
      state.serialize();
      that.userHasInteracted();
      that.heatMap.updateTestPoints(state.showTestData ? that.testData : []);
    });
    // Check/uncheck the checkbox according to the current state.
    showTestData.property('checked', state.showTestData);

    const discretize = d3.select('#discretize').on('change', function(){
      state.discretize =  d3.select(this).attr('checked');
      state.serialize();
      that.userHasInteracted();
      that.updateUI();
    });
    //Check/uncheck the checbox according to the current state.
    discretize.property('checked', state.discretize);

    const percTrain = d3.select('#percTrainData').on('input', function(){
      let value = d3.select(this).property('value');
      state.percTrainData = value;
      d3.select("label[for='percTrainData'].value").text(value);
      that.generateData();
      that.parametersChanged = true;
      that.reset();
    });
    percTrain.property('value', state.percTrainData);

    d3.select("label[for='percTrainData'] .value").text(state.percTrainData);

    const noise = d3.select('#noise').on('input', function(){
      let value = d3.select(this).property('value');
      state.noise = value;
      d3.select("label[for='noise'] .value").text(value);
      that.generateData();
      that.parametersChanged = true;
      that.reset();
    });
    noise.property('value', state.noise);
    d3.select("label[for='noise'] .value").text(state.noise);

    const batchSize = d3.select('#batchSize').on('input', function(){
      let value = d3.select(this).property('value');
      state.batchSize = value;
      d3.select("label[for='batchSize'] .value").text(value);
      that.parametersChanged = true;
      that.reset();
    });
    batchSize.property('value', state.batchSize);
    d3.select("label[for='batchSize'] .value").text(state.batchSize);

    const activationDropdown = d3.select('#activations').on('change', function(){
      let value = d3.select(this).property('value');
      state.activation = activations[value];
      that.parametersChanged = true;
      that.reset();
    });
    activationDropdown.property('value',
        getKeyFromValue(activations, state.activation));

    const learningRate = d3.select('#learningRate').on('change', function(){
      let value = d3.select(this).property('value');
      state.learningRate = value;
      state.serialize();
      that.userHasInteracted();
      that.parametersChanged = true;
    });
    learningRate.property('value', state.learningRate);

    const regularDropdown = d3.select('#regularizations').on('change',
        function(){
      let value = d3.select(this).property('value');
      state.regularization = regularizations[value];
      that.parametersChanged = true;
      that.reset();
    });
    regularDropdown.property('value',
        getKeyFromValue(regularizations, state.regularization));

    const regularRate = d3.select('#regularRate').on('change', function(){
      let value = d3.select(this).property('value');
      state.regularizationRate = value;
      that.parametersChanged = true;
      that.reset();
    });
    regularRate.property('value', state.regularizationRate);

    const problem = d3.select('#problem').on('change', function(){
      let value = d3.select(this).property('value');
      state.problem = problems[value];
      that.generateData();
      that.drawDatasetThumbnails();
      that.parametersChanged = true;
      that.reset();
    });
    problem.property('value', getKeyFromValue(problems, state.problem));

    // Add scale to the gradient color map.
    const x = d3.scaleLinear().domain([-1, 1]).range([0, 144]);
    const xAxis = axisBottom(x)
      .tickValues([-1, 0, 1])
      .tickFormat(d3.format('d'));
    d3.select('#colormap g.core').append('g')
      .attr('class', 'x axis')
      .attr('transform', 'translate(0,10)')
      .call(xAxis);

    // Listen for css-responsive changes and redraw the svg network.

    window.addEventListener('resize', () => {
      const newWidth = document.querySelector('#main-part')
          .getBoundingClientRect().width;
      let mainWidth = 500;
      if (newWidth !== mainWidth) {
        mainWidth = newWidth;
        that.drawNetwork(that.network);
        that.updateUI(true);
      }
    });

    // Hide the text below the visualization depending on the URL.
    if (state.hideText) {
      d3.select('#article-text').style('display', 'none');
      d3.select('div.more').style('display', 'none');
      d3.select('header').style('display', 'none');
    }


    that.drawDatasetThumbnails();
    that.initTutorial();
    that.makeGUI();
    that.generateData(true);
    that.reset(true);
    // that.hideControls();
  }

  /** Plays/pauses the player. */
  private playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (this.iter === 0) {
        this.simulationStarted();
      }
      this.play();
    }
  }

  private onPlayPause(isPlaying: boolean) {
    d3.select('#play-pause-button').classed('playing', isPlaying);
  }

  play() {
    this.pause();
    this.isPlaying = true;
    this.onPlayPause(this.isPlaying);
    // if (this.callback) {
    //   this.callback(this.isPlaying);
    // }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    this.onPlayPause(this.isPlaying);
    // if (this.callback) {
    //   this.callback(this.isPlaying);
    // }
  }

  private start(localTimerIndex: number) {
    let that = this;
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true;  // Done.
      }
      that.oneStep();
      return false;  // Not done.
    }, 0);
  }

  private makeGUI() {
    let that = this;
    d3.select('#reset-button').on('click', () => {
      that.reset();
      that.userHasInteracted();
      d3.select('#play-pause-button');
    });
  }

  private updateBiasesUI(network: nn.Node[][]) {
    let that = this;
    nn.forEachNode(network, true, node => {
      d3.select(`rect#bias-${node.id}`).style('fill', that.colorScale(node.bias));
    });
  }

  private updateWeightsUI(network: nn.Node[][], container: d3.Selection<d3.BaseType, {}, HTMLElement, any>) {
    let that = this;
    for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
      const currentLayer = network[layerIdx];
      // Update all the nodes in this layer.
      for (let i = 0; i < currentLayer.length; i++) {
        const node = currentLayer[i];
        for (let j = 0; j < node.inputLinks.length; j++) {
          const link = node.inputLinks[j];
          container.select(`#link${link.source.id}-${link.dest.id}`)
              .style('stroke-dashoffset',  -that.iter / 3)
              .style('stroke-width', that.linkWidthScale(Math.abs(link.weight)))
              .style('stroke', that.colorScale(link.weight))
              .datum(link);
        }
      }
    }
  }

  private drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
      container: d3.Selection<d3.BaseType, {}, HTMLElement, any>, node?: nn.Node) {
    let state = this.state;
    let that = this;
    const x = cx - PlaygroundComponent.RECT_SIZE / 2;
    const y = cy - PlaygroundComponent.RECT_SIZE / 2;

    const nodeGroup = container.append('g')
      .attr('class', 'node')
      .attr( 'id', `node${nodeId}`)
      .attr( 'transform', `translate(${x},${y})`);

    // Draw the main rectangle.
    nodeGroup.append('rect')
      .attr('x' , '0')
      .attr('y', '0')
      .attr('width', PlaygroundComponent.RECT_SIZE)
      .attr('height', PlaygroundComponent.RECT_SIZE);
    const activeOrNotClass = state[nodeId] ? 'active' : 'inactive';
    if (isInput) {
      const label = PlaygroundComponent.INPUTS[nodeId].label != null ?
        PlaygroundComponent.INPUTS[nodeId].label : nodeId;
      // Draw the input label.
      const text = nodeGroup.append('text')
      .attr('class', 'main-label')
      .attr('x', -10)
      .attr('y', PlaygroundComponent.RECT_SIZE / 2)
      .attr('text-anchor', 'end');
      if (/[_^]/.test(label)) {
        const myRe = /(.*?)([_^])(.)/g;
        let myArray;
        let lastIndex;
        while ((myArray = myRe.exec(label)) != null) {
          lastIndex = myRe.lastIndex;
          const prefix = myArray[1];
          const sep = myArray[2];
          const suffix = myArray[3];
          if (prefix) {
            text.append('tspan').text(prefix);
          }
          text.append('tspan')
          .attr('baseline-shift', sep === '_' ? 'sub' : 'super')
          .style('font-size', '9px')
          .text(suffix);
        }
        if (label.substring(lastIndex)) {
          text.append('tspan').text(label.substring(lastIndex));
        }
      } else {
        text.append('tspan').text(label);
      }
      nodeGroup.classed(activeOrNotClass, true);
    }
    if (!isInput) {
      // Draw the node's bias.
      nodeGroup.append('rect')
        .attr('id', `bias-${nodeId}`)
        .attr('x', -PlaygroundComponent.BIAS_SIZE - 2)
        .attr('y', PlaygroundComponent.RECT_SIZE - PlaygroundComponent.BIAS_SIZE + 3)
        .attr('width', PlaygroundComponent.BIAS_SIZE)
        .attr('height', PlaygroundComponent.BIAS_SIZE)
        .on('mouseenter', function() {
          that.updateHoverCard(HoverType.BIAS, node, d3.mouse(container.node() as any));
        }).on('mouseleave', function() {
          that.updateHoverCard(null);
        });
    }

    // Draw the node's canvas.
    const div = d3.select('#network').insert('div', ':first-child')
      .attr('id', `canvas-${nodeId}`)
      .attr( 'class', 'canvas')
      .style( 'position', 'absolute')
      .style( 'left', `${x + 3}px`)
      .style( 'top', `${y + 3}px`)
      .on('mouseenter', function() {
        that.selectedNodeId = nodeId;
        div.classed('hovered', true);
        nodeGroup.classed('hovered', true);
        that.updateDecisionBoundary(that.network, false);
        that.heatMap.updateBackground(that.boundary[nodeId], state.discretize);
      })
      .on('mouseleave', function() {
        that.selectedNodeId = null;
        div.classed('hovered', false);
        nodeGroup.classed('hovered', false);
        that.updateDecisionBoundary(that.network, false);
        that.heatMap.updateBackground(that.boundary[nn.getOutputNode(that.network).id],
            state.discretize);
      });
    if (isInput) {
      div.on('click', function() {
        state[nodeId] = !state[nodeId];
        that.parametersChanged = true;
        that.reset();
      });
      div.style('cursor', 'pointer');
    }
    if (isInput) {
      div.classed(activeOrNotClass, true);
    }
    const nodeHeatMap = new HeatMap(PlaygroundComponent.RECT_SIZE, PlaygroundComponent.DENSITY / 10, that.xDomain,
      that.xDomain, div, {noSvg: true});
    div.datum({heatmap: nodeHeatMap, id: nodeId});

  }

  // Draw network
  private drawNetwork(network: nn.Node[][]): void {
    let that = this;
    const svg = d3.select('#svg');
    // Remove all svg elements.
    svg.select('g.core').remove();
    // Remove all div elements.
    d3.select('#network').selectAll('div.canvas').remove();
    d3.select('#network').selectAll('div.plus-minus-neurons').remove();

    // Get the width of the svg container.
    const padding = 3;
    const co = d3.select('.column.output').node() as HTMLDivElement;
    const cf = d3.select('.column.features').node() as HTMLDivElement;
    const width = co.offsetLeft - cf.offsetLeft;
    svg.attr('width', width);

    // Map of all node coordinates.
    const node2coord: {[id: string]: {cx: number, cy: number}} = {};
    const container = svg.append('g')
      .classed('core', true)
      .attr('transform', 'translate('+padding+','+padding+')');
    // Draw the network layer by layer.
    const numLayers = network.length;
    const featureWidth = 118;
    const layerScale = d3.scalePoint<Number>()
        .domain(  (d3.range(1, numLayers - 1)) ) // .map(String)
        // .rangePoints([featureWidth, width - RECT_SIZE], 0.7);
        .range([featureWidth, width - PlaygroundComponent.RECT_SIZE])
        .padding(0.7)
        ;
    const nodeIndexScale = (nodeIndex: number) => nodeIndex * (PlaygroundComponent.RECT_SIZE + 25);


    const calloutThumb = d3.select('.callout.thumbnail').style('display', 'none');
    const calloutWeights = d3.select('.callout.weights').style('display', 'none');
    let idWithCallout = null;
    let targetIdWithCallout = null;

    // Draw the input layer separately.
    let cx = PlaygroundComponent.RECT_SIZE / 2 + 50;
    const nodeIds = Object.keys(PlaygroundComponent.INPUTS);
    let maxY = nodeIndexScale(nodeIds.length);
    nodeIds.forEach((nodeId, i) => {
      const cy = nodeIndexScale(i) + PlaygroundComponent.RECT_SIZE / 2;
      node2coord[nodeId] = {cx, cy};
      that.drawNode(cx, cy, nodeId, true, container);
    });

    // Draw the intermediate layers.
    for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
      const numNodes = network[layerIdx].length;
      const cx = layerScale(layerIdx) + PlaygroundComponent.RECT_SIZE / 2;
      maxY = Math.max(maxY, nodeIndexScale(numNodes));
      that.addPlusMinusControl(layerScale(layerIdx), layerIdx);
      for (let i = 0; i < numNodes; i++) {
        const node = network[layerIdx][i];
        const cy = nodeIndexScale(i) + PlaygroundComponent.RECT_SIZE / 2;
        node2coord[node.id] = {cx, cy};
        that.drawNode(cx, cy, node.id, false, container, node);

        // Show callout to thumbnails.
        const numNodes = network[layerIdx].length;
        const nextNumNodes = network[layerIdx + 1].length;
        if (idWithCallout == null &&
            i === numNodes - 1 &&
            nextNumNodes <= numNodes) {
          calloutThumb
          .style('display', 'none')   // null
          .style('top', '${20 + 3 + cy}px')
          .style('left', '${cx}px');

          idWithCallout = node.id;
        }

        // Draw links.
        for (let j = 0; j < node.inputLinks.length; j++) {
          const link = node.inputLinks[j];
          const path: SVGPathElement = that.drawLink(link, node2coord, network,
              container, j === 0, j, node.inputLinks.length).node() as any;
          // Show callout to weights.
          const prevLayer = network[layerIdx - 1];
          const lastNodePrevLayer = prevLayer[prevLayer.length - 1];
          if (targetIdWithCallout == null &&
              i === numNodes - 1 &&
              link.source.id === lastNodePrevLayer.id &&
              (link.source.id !== idWithCallout || numLayers <= 5) &&
              link.dest.id !== idWithCallout &&
              prevLayer.length >= numNodes) {
            const midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
            calloutWeights
            .style('display', 'null')
            .style('top:', `${midPoint.y + 5}px`)
            .style('left', `${midPoint.x + 3}px`);

            targetIdWithCallout = link.dest.id;
          }
        }
      }
    }

    // Draw the output node separately.
    cx = width + PlaygroundComponent.RECT_SIZE / 2;
    const node = network[numLayers - 1][0];
    const cy = nodeIndexScale(0) + PlaygroundComponent.RECT_SIZE / 2;
    node2coord[node.id] = {cx, cy};
    // Draw links.
    for (let i = 0; i < node.inputLinks.length; i++) {
      const link = node.inputLinks[i];
      that.drawLink(link, node2coord, network, container, i === 0, i,
          node.inputLinks.length);
    }
    // Adjust the height of the svg.
    svg.attr('height', maxY);

    // Adjust the height of the features column.
    const height = Math.max(
      that.getRelativeHeight(calloutThumb),
      that.getRelativeHeight(calloutWeights),
      that.getRelativeHeight(d3.select('#network'))
    );
    d3.select('.column.features').style('height', height + 'px');
  }

  private getRelativeHeight(selection: d3.Selection<d3.BaseType, {}, HTMLElement, any>) {
    const node = selection.node() as HTMLAnchorElement;
    return node.offsetHeight + node.offsetTop;
  }

  private addPlusMinusControl(x: number, layerIdx: number) {
    let that = this;
    let state = this.state;
    const div = d3.select('#network').append('div')
      .classed('plus-minus-neurons', true)
      .style('left', `${x - 10}px`);

    const i = layerIdx - 1;
    const firstRow = div.append('div').attr('class', `ui-numNodes${layerIdx}`);
    firstRow.append('button')
        .attr('class', 'mdl-button mdl-js-button mdl-button--icon')
        .on('click', () => {
          const numNeurons = state.networkShape[i];
          if (numNeurons >= 8) {
            return;
          }
          state.networkShape[i]++;
          that.parametersChanged = true;
          that.reset();
        })
      .append('i')
        .attr('class', 'material-icons')
        .text('add');

    firstRow.append('button')
        .attr('class', 'mdl-button mdl-js-button mdl-button--icon')
        .on('click', () => {
          const numNeurons = state.networkShape[i];
          if (numNeurons <= 1) {
            return;
          }
          state.networkShape[i]--;
          that.parametersChanged = true;
          that.reset();
        })
      .append('i')
        .attr('class', 'material-icons')
        .text('remove');

    const suffix = state.networkShape[i] > 1 ? 's' : '';
    div.append('div').text(
      state.networkShape[i] + ' neuron' + suffix
    );
  }

  private updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link,
      coordinates?: [number, number]) {
    const hovercard = d3.select('#hovercard');
    let that = this;
    if (type == null) {
      hovercard.style('display', 'none');
      d3.select('#svg').on('click', null);
      return;
    }
    d3.select('#svg').on('click', () => {
      hovercard.select('.value').style('display', 'none');
      const input = hovercard.select('input');
      input.style('display', null);
      input.on('input', function(){
        let value = d3.select(this).property('value');
        if (value != null && value !== '') {
          if (type === HoverType.WEIGHT) {
            (nodeOrLink as nn.Link).weight = +value;
          } else {
            (nodeOrLink as nn.Node).bias = +value;
          }
          that.updateUI();
        }
      });
      input.on('keypress', () => {
        if ((d3.event as any).keyCode === 13) {
          that.updateHoverCard(type, nodeOrLink, coordinates);
        }
      });
      (input.node() as HTMLInputElement).focus();
    });

    const value = (type === HoverType.WEIGHT) ?
      (nodeOrLink as nn.Link).weight :
      (nodeOrLink as nn.Node).bias;

    const name = (type === HoverType.WEIGHT) ? 'Weight' : 'Bias';
    hovercard.style('left', `${coordinates[0] + 20}px`)
    .style('top', `${coordinates[1]}px`)
    .style('display', 'block');
    hovercard.select('.type').text(name);
    hovercard.select('.value')
      .style('display', null)
      .text(value.toPrecision(2));
    hovercard.select('input')
      .property('value', value.toPrecision(2))
      .style('display', 'none');
  }

  private drawLink(
      input: nn.Link, node2coord: {[id: string]: {cx: number, cy: number}},
      network: nn.Node[][], container: d3.Selection<d3.BaseType, {}, HTMLElement, any>,
      isFirst: boolean, index: number, length: number) {

    let that = this;
    const line = container.insert('path', ':first-child');
    const source = node2coord[input.source.id];
    const dest = node2coord[input.dest.id];



    const d = {
      source: {
        y: source.cx + PlaygroundComponent.RECT_SIZE / 2 + 2,
        x: source.cy
      },
      target: {
        y: dest.cx - PlaygroundComponent.RECT_SIZE / 2,
        x: dest.cy + ((index - (length - 1) / 2) / length) * 12
      }
    };
    //const diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);
    var dd = "M" + d.source.y + "," + d.source.x
          + "C" + (d.source.y + d.target.y) / 2 + "," + d.source.x
          + " " + (d.source.y + d.target.y) / 2 + "," + d.target.x
          + " " + d.target.y + "," + d.target.x;
    line.attr('marker-start', 'url(#markerArrow)')
    line.attr('class', 'link')
    line.attr('id', 'link' + input.source.id + '-' + input.dest.id)
      // d: diagonal(datum, 0)
      .attr("d", dd);

      //// d3.linkHorizontal().
      // x(function(d) { return d.y; })
      // .y(function(d) { return d.x; }));

    // Add an invisible thick link that will be used for
    // showing the weight value on hover.
    container.append('path')
      // .attr('d', diagonal(datum, 0))
      // .attr("d", d3.linkHorizontal())
      .attr("d", dd)
      .attr('class', 'link-hover')
      .on('mouseenter', function(e){
        //that.updateHoverCard(HoverType.WEIGHT, input, d3.mouse(this));
      }).on('mouseleave', function() {
        that.updateHoverCard(null);
      });
    return line;
  }

  /**
   * Given a neural network, it asks the network for the output (prediction)
   * of every node in the network using inputs sampled on a square grid.
   * It returns a map where each key is the node ID and the value is a square
   * matrix of the outputs of the network for each input in the grid respectively.
   */
  private updateDecisionBoundary(network: nn.Node[][], firstTime: boolean) {
    let that = this;
    if (firstTime) {
      that.boundary = {};
      nn.forEachNode(network, true, node => {
        that.boundary[node.id] = new Array(PlaygroundComponent.DENSITY);
      });
      // Go through all predefined inputs.
      for (const nodeId in PlaygroundComponent.INPUTS) {
        that.boundary[nodeId] = new Array(PlaygroundComponent.DENSITY);
      }
    }
    const xScale = d3.scaleLinear().domain([0, PlaygroundComponent.DENSITY - 1]).range(that.xDomain);
    const yScale = d3.scaleLinear().domain([PlaygroundComponent.DENSITY - 1, 0]).range(that.xDomain);

    let i = 0, j = 0;
    for (i = 0; i < PlaygroundComponent.DENSITY; i++) {
      if (firstTime) {
        nn.forEachNode(network, true, node => {
          that.boundary[node.id][i] = new Array(PlaygroundComponent.DENSITY);
        });
        // Go through all predefined inputs.
        for (const nodeId in PlaygroundComponent.INPUTS) {
          that.boundary[nodeId][i] = new Array(PlaygroundComponent.DENSITY);
        }
      }
      for (j = 0; j < PlaygroundComponent.DENSITY; j++) {
        // 1 for points inside the circle, and 0 for points outside the circle.
        const x = xScale(i);
        const y = yScale(j);
        const input = that.constructInput(x, y);
        nn.forwardProp(network, input);
        nn.forEachNode(network, true, node => {
          that.boundary[node.id][i][j] = node.output;
        });
        if (firstTime) {
          // Go through all predefined inputs.
          for (const nodeId in PlaygroundComponent.INPUTS) {
            that.boundary[nodeId][i][j] = PlaygroundComponent.INPUTS[nodeId].f(x, y);
          }
        }
      }
    }
  }

  private getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
    let loss = 0;
    let that = this;
    for (let i = 0; i < dataPoints.length; i++) {
      const dataPoint = dataPoints[i];
      const input = that.constructInput(dataPoint.x, dataPoint.y);
      const output = nn.forwardProp(network, input);
      loss += nn.Errors.SQUARE.error(output, dataPoint.label);
    }
    return loss / dataPoints.length;
  }

  private updateUI(firstStep = false) {
    let state = this.state;
    let that = this;
    // Update the links visually.
    that.updateWeightsUI(that.network, d3.select('g.core'));
    // Update the bias values visually.
    that.updateBiasesUI(that.network);
    // Get the decision boundary of the network.
    that.updateDecisionBoundary(that.network, firstStep);
    const selectedId = that.selectedNodeId != null ?
        that.selectedNodeId : nn.getOutputNode(that.network).id;
    that.heatMap.updateBackground(that.boundary[selectedId], state.discretize);

    // Update all decision boundaries.
    d3.select('#network').selectAll('div.canvas')
        .each(function(data: {heatmap: HeatMap, id: string}) {
      data.heatmap.updateBackground(reduceMatrix(that.boundary[data.id], 10),
          state.discretize);
    });

    function zeroPad(n: number): string {
      const pad = '000000';
      return (pad + n).slice(-pad.length);
    }

    function addCommas(s: string): string {
      return s.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    }

    function humanReadable(n: number): string {
      return n.toFixed(3);
    }

    // Update loss and iteration number.
    d3.select('#loss-train').text(humanReadable(that.lossTrain));
    d3.select('#loss-test').text(humanReadable(that.lossTest));
    d3.select('#iter-number').text(addCommas(zeroPad(that.iter)));
    that.lineChart.addDataPoint([that.lossTrain, that.lossTest]);
  }

  private constructInputIds(): string[] {
    let state = this.state;
    const result: string[] = [];
    for (const inputName in PlaygroundComponent.INPUTS) {
      if (state[inputName]) {
        result.push(inputName);
      }
    }
    return result;
  }

  private constructInput(x: number, y: number): number[] {
    let state = this.state;
    const input: number[] = [];
    for (const inputName in PlaygroundComponent.INPUTS) {
      if (state[inputName]) {
        input.push(PlaygroundComponent.INPUTS[inputName].f(x, y));
      }
    }
    return input;
  }

  private oneStep(): void {
    let state = this.state;
    let that = this;
    that.iter++;
    that.trainData.forEach((point, i) => {
      const input = that.constructInput(point.x, point.y);
      nn.forwardProp(that.network, input);
      nn.backProp(that.network, point.label, nn.Errors.SQUARE);
      if ((i + 1) % state.batchSize === 0) {
        nn.updateWeights(that.network, state.learningRate, state.regularizationRate);
      }
    });
    // Compute the loss.
    that.lossTrain = that.getLoss(that.network, that.trainData);
    that.lossTest = that.getLoss(that.network, that.testData);
    that.updateUI();
  }

  private getOutputWeights(network: nn.Node[][]): number[] {
    const weights: number[] = [];
    for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
      const currentLayer = network[layerIdx];
      for (let i = 0; i < currentLayer.length; i++) {
        const node = currentLayer[i];
        for (let j = 0; j < node.outputs.length; j++) {
          const output = node.outputs[j];
          weights.push(output.weight);
        }
      }
    }
    return weights;
  }

  private reset(onStartup= false) {
    let state = this.state;
    let that = this;
    that.lineChart.reset();
    state.serialize();
    if (!onStartup) {
      that.userHasInteracted();
    }
    that.pause();

    const suffix = state.numHiddenLayers !== 1 ? 's' : '';
    d3.select('#layers-label').text('Hidden layer' + suffix);
    d3.select('#num-layers').text(state.numHiddenLayers);

    // Make a simple network.
    that.iter = 0;
    const numInputs = that.constructInput(0 , 0).length;
    const shape = [numInputs].concat(state.networkShape).concat([1]);
    const outputActivation = (state.problem === Problem.REGRESSION) ?
        nn.Activations.LINEAR : nn.Activations.TANH;
    that.network = nn.buildNetwork(shape, state.activation, outputActivation,
        state.regularization, that.constructInputIds(), state.initZero);
    that.lossTrain = that.getLoss(that.network, that.trainData);
    that.lossTest = that.getLoss(that.network, that.testData);
    that.drawNetwork(that.network);
    that.updateUI(true);
  };

  private initTutorial() {
    let state = this.state;
    if (state.tutorial == null || state.tutorial === '' || state.hideText) {
      return;
    }
    // Remove all other text.
    d3.selectAll('article div.l--body').remove();
    const tutorial = d3.select('article').append('div')
      .attr('class', 'l--body');
    // Insert tutorial text.
    d3.html(`tutorials/${state.tutorial}.html`, (err, htmlFragment) => {
      if (err) {
        throw err;
      }
      const node = tutorial.node() as any;
      node.appendChild(htmlFragment);
      // If the tutorial has a <title> tag, set the page title to that.
      const title = tutorial.select('title');
      if (title.size()) {
        d3.select('header h1')
        .style( 'margin-top', '20px')
        .style( 'margin-bottom', '20px')
        .text(title.text());
        document.title = title.text();
      }
    });
  }

  private renderThumbnail(canvas, dataGenerator) {
    const w = 100;
    const h = 100;
    let that = this;
    canvas.setAttribute('width', w);
    canvas.setAttribute('height', h);
    const context = canvas.getContext('2d');
    const data = dataGenerator(200, 0);
    data.forEach(function(d) {
      context.fillStyle = that.colorScale(d.label);
      context.fillRect(w * (d.x + 6) / 12, h * (d.y + 6) / 12, 4, 4);
    });
    d3.select(canvas.parentNode).style('display', null);
  }

  private drawDatasetThumbnails() {
    let state = this.state;
    d3.selectAll('.dataset').style('display', 'none');

    if (state.problem === Problem.CLASSIFICATION) {
      for (const dataset in datasets) {
        const canvas: any =
            document.querySelector(`canvas[data-dataset=${dataset}]`);
        const dataGenerator = datasets[dataset];
        this.renderThumbnail(canvas, dataGenerator);
      }
    }
    if (state.problem === Problem.REGRESSION) {
      for (const regDataset in regDatasets) {
        const canvas: any =
            document.querySelector(`canvas[data-regDataset=${regDataset}]`);
        const dataGenerator = regDatasets[regDataset];
        this.renderThumbnail(canvas, dataGenerator);
      }
    }
  }

  private hideControls() {
    let state = this.state;
    let that = this;
    // Set display:none to all the UI elements that are hidden.
    const hiddenProps = state.getHiddenProps();
    hiddenProps.forEach(prop => {
      const controls = d3.selectAll(`.ui-${prop}`);
      if (controls.size() === 0) {
        console.warn(`0 html elements found with class .ui-${prop}`);
      }
      controls.style('display', 'none');
    });

    // Also add checkbox for each hidable control in the 'use it in classrom'
    // section.
    const hideControls = d3.select('.hide-controls');
    PlaygroundComponent.HIDABLE_CONTROLS.forEach(([text, id]) => {
      const label = hideControls.append('label')
        .attr('class', 'mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect');
      const input = label.append('input')
        .attr('type', 'checkbox')
        .attr('class', 'mdl-checkbox__input');
      if (hiddenProps.indexOf(id) === -1) {
        input.attr('checked', 'true');
      }
      input.on('change', () => {
        // state.setHideProperty(id, !this.checked);
        state.serialize();
        that.userHasInteracted();
        d3.select('.hide-controls-link')
          .attr('href', window.location.href);
      });
      label.append('span')
        .attr('class', 'mdl-checkbox__label label')
        .text(text);
    });
    d3.select('.hide-controls-link')
      .attr('href', window.location.href);
  }

  private generateData(firstTime = false) {
    let state = this.state;
    let that = this;
    if (!firstTime) {
      // Change the seed.
      state.seed = Math.random().toFixed(5);
      state.serialize();
      that.userHasInteracted();
      }
      //// Sets Math.random to a PRNG initialized using the given explicit seed.
      //TODO see if this works for TS
      //Fixme
    //_.seedrandom(state.seed);
    const numSamples = (state.problem === Problem.REGRESSION) ?
        PlaygroundComponent.NUM_SAMPLES_REGRESS : PlaygroundComponent.NUM_SAMPLES_CLASSIFY;
    const generator = state.problem === Problem.CLASSIFICATION ?
        state.dataset : state.regDataset;
    const data = generator(numSamples, state.noise / 100);
    // Shuffle the data in-place.
    shuffle(data);
    // Split into train and test data.
    const splitIndex = Math.floor(data.length * state.percTrainData / 100);
    that.trainData = data.slice(0, splitIndex);
    that.testData = data.slice(splitIndex);
    that.heatMap.updatePoints(that.trainData);
    that.heatMap.updateTestPoints(state.showTestData ? that.testData : []);
  }

  private userHasInteracted() {
    let that = this;
    let state = this.state;
    if (!that.firstInteraction) {
      return;
    }
    that.firstInteraction = false;
    let page = 'index';
    if (state.tutorial != null && state.tutorial !== '') {
      page = `/v/tutorials/${state.tutorial}`;
    }
    ga('set', 'page', page);
    ga('send', 'pageview', {'sessionControl': 'start'});
  }

  private simulationStarted() {
    let state = this.state;
    let that = this;
    ga('send', {
      hitType: 'event',
      eventCategory: 'Starting Simulation',
      eventAction: that.parametersChanged ? 'changed' : 'unchanged',
      eventLabel: state.tutorial == null ? '' : state.tutorial
    });
    that.parametersChanged = false;
  }

}
