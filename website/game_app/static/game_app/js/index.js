
const popular = document.querySelectorAll('#popular .grid article')
const rec_rel = document.querySelectorAll('#recently-released .grid article')

window.onload = () => {
    const length = 3;
    for(let i = 3; i < popular.length; i += 1)
        {
            popular[i].classList.add('none');
        }
    for(let i = 3; i < rec_rel.length; i += 1)
        {
            rec_rel[i].classList.add('none');
        }
}

const groups = document.querySelectorAll('.group')

const next = document.querySelectorAll('#categories button')[1]
const prev = document.querySelectorAll('#categories button')[0]

function swap_categories(direction)
{
    const length = groups.length;
    let id;
    groups.forEach(el => {
       if (el.classList.contains('active'))   
       {
            id = +el.dataset.id + direction;
            console.log(id);
            if (id < 0) id = length-1;
            if (id == length) id = 0;
            el.classList.remove('active')
       } 
    });  

    groups.forEach(el => {
        if (+el.dataset.id == id)
        {
            el.classList.add('active');
        }
    });
}

next.onclick = () =>
{
    swap_categories(1);
}

prev.onclick = () =>
{
    swap_categories(-1);
}

let popular_offset = {offset: 0};
let rel_offset = {offset: 0};

function swap_games(games, offset){
    offset.offset = (offset.offset + 1) % 2;

    const start = 0 + 3*offset.offset;

    games.forEach(el => {el.classList.add('none')});

    for (let i = start; i < start + 3 && i < games.length ; i += 1)
    {
        games[i].classList.remove('none');
//        console.log(games[i], offset.offset);
    }

}
