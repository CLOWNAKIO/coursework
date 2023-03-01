//const data = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ];
const labels = ["1","2","3","4","5","6","7","8","9","10"];
const canvas = document.getElementById('canvas'); 
  
// получаем указатель на контекст рисования
const c = canvas.getContext('2d'); 
c.font = "0.6rem Roboto, sans-serif "
// рисуем
c.fillStyle = "#202020"; 
c.fillRect(0,0,300,200);

const range = 20;
const per = 200 / Math.max(...data) / 1.2;
let start = 0;
let end = -5;
for(let i=0; i< data.length; i++) { 
  start = end + 10; //0 50 100
  end = start + range; // 40 90 140
  c.fillStyle = "#ba5e12";
  c.fillRect(start, 200, range, -data[i]*per); 
  c.fillStyle = "#DFDFDF";
  if (data[i] > 0 )
    c.fillText(data[i], start + 1, 200-data[i]*per - 5); //count
  c.fillText(labels[i], start + 4, 198); 
}

let progress_list = document.querySelector('aside ul').children;
const profile = progress_list[0];
const info = progress_list[1];
const library = progress_list[2];
const rec_list = progress_list[3];
const settings = progress_list[4];
const logout = progress_list[5];

profile.onclick = () => {scrollTo(0, 78);};
info.onclick = () => {scrollTo(0, 447);};
library.onclick = () => {scrollTo(0, 999);};
rec_list.onclick = () => {scrollTo(0, 2042);};
settings.onclick = () => {scrollTo(0, 2042);};
logout.onclick = () => {scrollTo(0, 2042);};

let posTop = 0;
let progress = innerHeight / document.documentElement.scrollHeight;
const progress_bar = document.querySelector('.progress-bar');
progress_bar.style.height = 0 + 'px';
window.onscroll = function() {
	posTop = (window.pageYOffset !== undefined) ? window.pageYOffset : (document.documentElement || document.body.parentNode || document.body).scrollTop;
	// console.log( posTop);
  progress = posTop / document.documentElement.scrollHeight
  progress_bar.style.height = progress * 160 + 'px';
}   // высота прокрута

// let scrollHeight = Math.max(
//   document.body.scrollHeight, document.documentElement.scrollHeight,
//   document.body.offsetHeight, document.documentElement.offsetHeight,
//   document.body.clientHeight, document.documentElement.clientHeight
// );    // высота документа
