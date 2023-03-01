const selectors = document.querySelectorAll('input[type="radio"]')
const labels = document.querySelectorAll('main label')
const lists = document.querySelectorAll('.content div')

selectors.forEach(el => el.onchange = () => {
labels.forEach(el => el.classList.toggle('selected'))
lists.forEach(el => el.classList.toggle('hidden'))
})