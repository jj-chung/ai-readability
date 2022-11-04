// run npm i puppeteer
// npm install --save csvtojson@latest

const puppeteer = require('puppeteer');
const fs = require('fs/promises');
var csv = require("csvtojson");

var count = 0;
var error_count = 0

const URL = 'http://www.roadtogrammar.com/textanalysis/';

function level_to_int(level) {
    if (level === 'A2') {
        return 1;
    }
    if (level === 'B1') {
        return 2;
    }
    if (level === 'B2') {
        return 3;
    }
    if (level === 'C1') {
        return 4;
    }
    if (level === 'C2') {
        return 5;
    }
}

// initalize the browser URL
async function initBrowser() {
    const browser = await puppeteer.launch();
    // const browser = await puppeteer.launch({ignoreDefaultArgs: ['--disable-extensions']})
    const page = await browser.newPage();
    await page.goto(URL);
    return page;
}


async function get_cefr_values(page, string_text) {
    // enter the text value
    await page.evaluate((string_text) => {document.getElementById('tx').value = string_text}, string_text);
    // press submit button
    await page.evaluate(() => {document.getElementById('butang1').click()});
    await page.waitForTimeout(700);

    let final_level = -1;

    try {
        // get the level rating
        const level_text = await page.evaluate(() => {
            const temp = document.getElementById("text3").innerText;
            return temp;
        });
        level = level_text.slice(-2);
        level_number = level_to_int(level);

        // get the partial color percentage
        const boxHTML = await page.evaluate((level_number) => {
            const queryId = `a${level_number}1`;
            const element = document.getElementById(queryId).outerHTML;
            return element;
        }, level_number);
        percent_add = boxHTML.slice(boxHTML.indexOf('width:') + 7, boxHTML.indexOf('%'));
        final_level = level_number + parseFloat(percent_add)/100 - 1; 
    } 
    catch(err) {
        console.log('ERROR t_t' + err);
        error_count += 1;
        await page.screenshot({path: `picture${count}.png`});

    } finally {
        // click new text button
        await page.evaluate(() => {document.getElementById('butang3').click()});
        await page.waitForTimeout(100);
        return final_level;
    }
}

async function run_cefr_bot (string_texts) {
    const page = await initBrowser();
    const finalDict = {};
    for (let dicts of string_texts) {

        const string_text = dicts.excerpt;
        cefr_lvl = await get_cefr_values(page, string_text);
        dicts.cefr_lvl = cefr_lvl;
        id = dicts.id;

        finalDict[id] = dicts;
        console.log(dicts);

        count += 1;
        console.log(count);
    
        // if (count > 4) { break; }
    }

    const jsonString = JSON.stringify(finalDict, null, 2);
    fs.writeFile('./cefrDATA.json', jsonString, err => {
        if (err) {
            console.log('Error writing file', err)
        } else {
            console.log('Successfully wrote file')
        }
    })

    console.log("LOOK YOU'RE DONE!");
    console.log(`Completed with ${error_count} errors`)
}

const string_texts = [];
csv()
  .fromFile('CLEAR_Corpus_6.01.csv')
  .then(function(jsonArrayObj){ //when parse finished, result will be emitted here.
    for (let i = 0; i < jsonArrayObj.length; i++) {
        id = jsonArrayObj[i].ID;
        excerpt = jsonArrayObj[i].Excerpt;
        bt_easiness= jsonArrayObj[i]['BT Easiness'];
        string_texts.push({
            id: id,
            excerpt: excerpt,
            bt_easiness: bt_easiness,
            cefr_lvl: ''
        });
    }
    run_cefr_bot(string_texts);
   })