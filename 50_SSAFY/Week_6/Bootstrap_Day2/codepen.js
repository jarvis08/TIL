var select = function(s) {
    return document.querySelector(s);
  },
  selectAll = function(s) {
    return document.querySelectorAll(s);
  },
    colorArray = ["#EF476F", "#FFD166", "#06D6A0", "#118AB2", "#073B4C"],
    playArray = ["P", "L", "A2", "Y"],
    hardArray = ["H", "A", "R", "D"],
    offset = 0.15, duration = 1, mainTl = new TimelineMax({repeat: -1});
  

TweenMax.set('svg', {
  visibility: 'visible'
})

var ease1 = CustomEase.create("custom", "M0,0 C0.692,0.098 0.304,0.898 1,1");

TweenLite.defaultEase = ease1; 

function createStartAnimation() {
 let num = 4;
 for(var i = 0; i < num; i++) {
  let playTl = new TimelineMax();
  let playLetter = select('#'+playArray[i] + '_Start');
  let hardLetter = select('#'+hardArray[i] + '_End');
  playTl.to(playLetter, duration, {
   morphSVG:select('#'+playArray[i] + '_End'),
   repeat: -1,
   yoyo: true
  })
   .to(hardLetter, duration, {
   morphSVG:select('#'+hardArray[i] + '_Start'),
   repeat: -1,
   yoyo: true
  }, 0)

  mainTl.add([playTl], i * offset)
 }
}

createStartAnimation()

mainTl.to('#gradPattern', 3, {
  attr:{
    x:800
  },
  ease:Linear.easeNone,
 repeat: -1
}, 0)

//TweenMax.globalTimeScale(0.5)
