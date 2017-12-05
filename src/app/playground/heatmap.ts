import * as d3 from 'd3';
import { Axis, axisBottom, axisRight } from 'd3-axis'
import { Example2D } from './dataset';
import { select, Selection } from 'd3-selection';


// import * as d3Scale from 'd3-scale';
// import * as d3Shape from 'd3-shape';
// import * as d3Array from 'd3-array';
// import * as d3Axis from 'd3-axis';

export interface HeatMapSettings {
  [key: string]: any;
  showAxes?: boolean;
  noSvg?: boolean;
}

/** Number of different shades (colors) when drawing a gradient heatmap */
const NUM_SHADES = 30;

/**
 * Draws a heatmap using canvas. Used for showing the learned decision
 * boundary of the classification algorithm. Can also draw data points
 * using an svg overlayed on top of the canvas heatmap.
 */
export class HeatMap {
  private settings: HeatMapSettings = {
    showAxes: false,
    noSvg: false
  };


  private xScale: d3.ScaleContinuousNumeric<number, number>;
  private yScale: d3.ScaleContinuousNumeric<number, number>;
  private numSamples: number;
  private color: d3.ScaleQuantize<string>;
  private canvas: d3.Selection<d3.BaseType, {}, HTMLElement, any>;
  private svg: d3.Selection<d3.BaseType, {}, HTMLElement, any>;
  // private canvas: d3.Selection<SVGElement, {}, HTMLElement, any>;
  // private svg: d3.Selection<SVGElement, {}, HTMLElement, any>;


  constructor(
    width: number, numSamples: number, xDomain: [number, number],
    yDomain: [number, number], container: d3.Selection<d3.BaseType, {}, HTMLElement, any>,
    userSettings?: HeatMapSettings) {
    this.numSamples = numSamples;
    const height = width;
    const padding = userSettings.showAxes ? 20 : 0;

    if (userSettings != null) {
      // overwrite the defaults with the user-specified settings.
      for (const prop in userSettings) {
        if (userSettings.hasOwnProperty(prop)) {
          this.settings[prop] = userSettings[prop];
        }
      }
    }

    this.xScale = d3.scaleLinear()
      .domain(xDomain)
      .range([0, width - 2 * padding]);

    this.yScale = d3.scaleLinear()
      .domain(yDomain)
      .range([height - 2 * padding, 0]);

    // Get a range of colors.
    const tmpScale = d3.scaleLinear<string>()
      .domain([0, .5, 1])
      .range(['#f59322', '#e8eaeb', '#0877bd'])
      .clamp(true);
    // Due to numerical error, we need to specify
    // d3.range(0, end + small_epsilon, step)
    // in order to guarantee that we will have end/step entries with
    // the last element being equal to end.
    const colors = d3.range(0, 1 + 1E-9, 1 / NUM_SHADES).map(a => {
      return tmpScale(a);
    });
    this.color = d3.scaleQuantize<string>()
      .domain([-1, 1])
      .range(colors);
    // let cont = d3.select('bod');
    // cont.append("svg")
    // .attr(}width)
    // this.container = container;
    container.append('div')
      .style('width', width+'px')
      .style('height', height+'px')
      .style('position', 'relative')
      .style('top', padding+'px')
      .style('left', -padding+'px')
      ;

    // .style({
    //   width: "${width}px`,
    //   height: `${height}px`,
    //   position: 'relative',
    //   top: `-${padding}px`,
    //   left: `-${padding}px`
    // });
    this.canvas = container.append('canvas')
      .attr('width', numSamples)
      .attr('height', numSamples)
      .style('width', (width - 2 * padding) + 'px')
      .style('height', (height - 2 * padding) + 'px')
      .style('position', 'absolute')
      .style('top', padding+'px')
      .style('left', 0+'px');

    if (!this.settings.noSvg) {


      this.svg = container.append('svg')
        .attr('width', width)
        .attr('height', height)
        // Overlay the svg on top of the canvas.
        .style('position', 'absolute')
        .style('top', '0')
        .style('left', '0')
        .append('g')
        .attr('transform', 'translate('+0+','+padding+')');

      this.svg.append('g').attr('class', 'train');
      this.svg.append('g').attr('class', 'test');
    }

    if (this.settings.showAxes) {
      const xAxis = axisBottom(this.xScale);
      //  d3.svg.axis()
      //   .scale(this.xScale)
      //   .orient('bottom');

      const yAxis = axisRight(this.yScale);
      // d3.svg.axis()
      //   .scale(this.yScale)
      //   .orient('right');

      this.svg.append('g')
        .attr('class', 'x axis')
        .attr('transform', 'translate(0,'+(height - 2 * padding)+')')
        .call(xAxis);

      this.svg.append('g')
        .attr('class', 'y axis')
        .attr('transform', 'translate(' + (width - 2 * padding) + ',0)')
        .call(yAxis);
    }
  }

  updateTestPoints(points: Example2D[]): void {
    if (this.settings.noSvg) {
      throw Error('Can\'t add points since noSvg=true');
    }
    this.updateCircles(this.svg.select('g.test'), points);
  }

  updatePoints(points: Example2D[]): void {
    if (this.settings.noSvg) {
      throw Error('Can\'t add points since noSvg=true');
    }
    this.updateCircles(this.svg.select('g.train'), points);
  }

  updateBackground(data: number[][], discretize: boolean): void {
    const dx = data[0].length;
    const dy = data.length;

    if (dx !== this.numSamples || dy !== this.numSamples) {
      throw new Error(
        'The provided data matrix must be of size ' +
        'numSamples X numSamples');
    }

    // Compute the pixel colors; scaled by CSS.
    //    const context = (this.canvas.node() as HTMLCanvasElement).getContext('2d');
    const context = (this.canvas.node() as any as HTMLCanvasElement).getContext('2d');

    const image = context.createImageData(dx, dy);

    for (let y = 0, p = -1; y < dy; ++y) {
      for (let x = 0; x < dx; ++x) {
        let value = data[x][y];
        if (discretize) {
          value = (value >= 0 ? 1 : -1);
        }
        const c = d3.rgb(this.color(value));
        image.data[++p] = c.r;
        image.data[++p] = c.g;
        image.data[++p] = c.b;
        image.data[++p] = 160;
      }
    }
    context.putImageData(image, 0, 0);
  }

  private updateCircles(container: d3.Selection< d3.BaseType, {}, HTMLElement, any>, points: Example2D[]) {
    // Keep only points that are inside the bounds.
    const xDomain = this.xScale.domain();
    const yDomain = this.yScale.domain();
    points = points.filter(p => {
      return p.x >= xDomain[0] && p.x <= xDomain[1]
        && p.y >= yDomain[0] && p.y <= yDomain[1];
    });
    container.html('');

    // Attach data to initially empty selection.
    //const selection = container.selectAll('circle').data(points);

    // Insert elements to match length of points array.
    //selection.enter().append('circle').attr('r', 3);
    // Update points to be in the correct position.
    for(var i = 0; i < points.length; i++){
      container.append('circle')
              .attr('r', 3)
              .attr('cx', this.xScale(points[i].x))
              .attr('cy', this.yScale(points[i].y))
              .attr('fill', this.color(points[i].label));
    }
    // selection
    //   .attr('cx', (d) => this.xScale(d.x))
    //   .attr('cy', (d) => this.yScale(d.y))
    //   .style('fill', d => this.color(d.label));
    // .attr({
    //   cx: (d: Example2D) => this.xScale(d.x),
    //   cy: (d: Example2D) => this.yScale(d.y),
    // })
    // Remove points if the length has gone down.
    // selection.exit().remove();
  }
}  // Close class HeatMap.

export function reduceMatrix(matrix: number[][], factor: number): number[][] {
  if (matrix.length !== matrix[0].length) {
    throw new Error('The provided matrix must be a square matrix');
  }
  if (matrix.length % factor !== 0) {
    throw new Error('The width/height of the matrix must be divisible by ' +
      'the reduction factor');
  }
  const result: number[][] = new Array(matrix.length / factor);
  for (let i = 0; i < matrix.length; i += factor) {
    result[i / factor] = new Array(matrix.length / factor);
    for (let j = 0; j < matrix.length; j += factor) {
      let avg = 0;
      // Sum all the values in the neighborhood.
      for (let k = 0; k < factor; k++) {
        for (let l = 0; l < factor; l++) {
          avg += matrix[i + k][j + l];
        }
      }
      avg /= (factor * factor);
      result[i / factor][j / factor] = avg;
    }
  }
  return result;
}
