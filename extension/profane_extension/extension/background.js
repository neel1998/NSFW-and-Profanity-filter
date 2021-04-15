console.log('De-profaning the page');


const url = 'https://bdfb55801a13.ngrok.io/predict'


observers = {'detox': null}


const findProfanity = () => {

    document.querySelectorAll('p,h1,h2,h3,h4,h5,h6,a,span').forEach(x => {
      if (x !== null && !x.classList.value.includes("profane-checked")) {
        let text = x.textContent.trim();
        if (text.length > 0) {
                    
          let data = {'data': text};    

          $.post(url, data, function(data) {

            if (data['success'] === 'true') {

              let tox = parseFloat(data['target']);
              console.log(tox)
              if (tox > 0.5) {
                x.textContent = text[0] + "*".repeat(text.length - 1)
              }
              
            } else {

            }

          });

          x.classList.add("profane-checked");
      }
    }
  });

};


// const deProfane = textSection => {


//   findProfanity();

//   const mutationConfig = { attributes: false, childList: true, subtree: true };

//   if (observers['detox'] !== null) {

//     observers['detox'].observe(textSection, mutationConfig);

//   } else {     

//       const observer = new MutationObserver(() => {
//         console.log('in MutationObserver');
//         findProfanity();
//       });

//       observer.observe(textSection, mutationConfig);

//       observers['detox'] = observer;
//   };
  
// };


// const resetToxicity = () => {

//   console.log('in resetToxicity')

//   const selectors = ["#comment"];
//   const selection = selectors.map(sel => `${sel}.detoxified`).join(", ");

//   document.querySelectorAll(selection).forEach(x => {

//       let container = x.querySelector('#content');
//       let div = x.querySelector('#detox_container');

//       if (container !== null) {
//         container.setAttribute('style', 'background-color: #F9F9F9'); 
//       } 

//       if (div !== null) {
//         div.remove();
//         x.classList.remove("detoxified");
//       } 
         
//   });

// };




// const checkTextLoaded = () => {


//   // setTimeout(() => {
//     // const textSection = document.querySelector("p,h1,h2,h3,h4,h5,h6,a,span")
//     // console.log(textSection)

//     // if (textSection !== null) {

//     //   deProfane(textSection)
//     // }
//     // else checkTextLoaded();
//   // }, 1000);
// };


setInterval(() => {
 
  findProfanity()

}, 5000)

findProfanity()
// checkTextLoaded()

