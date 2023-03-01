window.onload = () => {
    let params = (new URL(document.location)).searchParams; 
    document.getElementById('name').value = (params.get("name")); 
    document.getElementById('start').value = params.get("start"); 
    if (params.get("start")){document.getElementById('start').classList.add('has-value')}
    document.getElementById('end').value = params.get("end"); 
    if (params.get("end")){document.getElementById('end').classList.add('has-value')}
    document.getElementById('platforms').value = params.get("platforms")
    document.getElementById('genres').value = params.get("genres")
    let platforms, genres, developers;
    platforms = params.get("platforms").split(',')
    platforms.pop()
    genres = params.get("genres").split(',')
    genres.pop()
    developers = params.get("platforms").split(',')
    developers.pop()
    // console.log(genres)
    // console.log(platforms)
    document.querySelectorAll('.list-container li[data-list="genres"]').forEach(el => { if (genres.includes(el.getAttribute('value'))){el.classList.add('selected')};  })
    document.querySelectorAll('.list-container li[data-list="platforms"]').forEach(el => {if (platforms.includes(el.getAttribute('value'))){el.classList.add('selected')} })
    document.querySelectorAll('.list-container li[data-list="developers"]').forEach(el => {if (developers.includes(el.getAttribute('value'))){el.classList.add('selected')} })
    // console.log(params.get("alp_asc"))
    if (params.get("alp_asc") == 'False')
    {
        document.querySelectorAll('.search-fields label').forEach(el => el.classList.toggle('selected'))
        document.getElementById('alp_desc').checked = true
    }

}

let list_btns = document.querySelectorAll('.choose-btn') 

list_btns.forEach(el => {
    el.addEventListener('click', () => {
        console.log()
        // el.classList.add('hidden');
        const ul = el.closest('div').childNodes[3]
        ul.classList.remove('hidden')
        ul.style.z_index = '1';
    })
})

// console.log('ku')

list_btns = document.querySelectorAll('.hide-btn') 

list_btns.forEach(el => {
    el.addEventListener('click', () => {

        // el.classList.add('hidden');
        const ul = el.closest('ul')
        ul.classList.add('hidden')
        ul.style.z_index = '0';
        // console.log(ul)
    })
})

list_btns = document.querySelectorAll('.list-container li')

list_btns.forEach(el => {
    el.addEventListener('click', () => {
        if (el.classList.contains('selected'))
        {
            let input_field = document.getElementById(el.dataset.list).value
            const value = el.getAttribute('value')
            const rep =  value + ','
            if (input_field.includes(value))
            {
                // document.getElementById(el.dataset.list).value = document.getElementById(el.dataset.list).value.replace(/el.getAttribute('value') + ','/, '')
                // console.log(document.getElementById(el.dataset.list).value)
                // console.log(document.getElementById(el.dataset.list).value.replace(rep,''))
                document.getElementById(el.dataset.list).value = document.getElementById(el.dataset.list).value.replace(rep,'')
            }
        }
        else
        {
            const rep =  el.getAttribute('value') + ','
            const input_field = document.getElementById(el.dataset.list).value
            document.getElementById(el.dataset.list).value += rep
        }
        el.classList.toggle('selected')

    })
})


document.getElementById('reset').onclick = () => 
{
    document.getElementById('start').classList.remove('has-value')
    document.getElementById('end').classList.remove('has-value')
    document.querySelectorAll('.list-container li.selected').forEach(el => { el.classList.remove('selected') })
}

const labels = document.querySelectorAll('.search-fields label')

labels.forEach(el => {
    el.addEventListener('click', () => {
        labels.forEach(elem => elem.classList.remove('selected'))
        el.classList.add('selected')
    })
})

window.onscroll = () => 
{
    if (window.pageYOffset > 500)
    {
        document.querySelector('.scroll-up .back').style.display = 'block'
    }
    else
    {
        document.querySelector('.scroll-up .back').style.display = 'none'
    }
}

