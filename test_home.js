const { chromium } = require('playwright');

(async () => {
  console.log('Starting playwright test...');
  const browser = await chromium.launch();
  const page = await browser.newPage();

  page.on('console', msg => {
    if (msg.type() === 'error') {
      console.log('BROWSER ERROR:', msg.text());
    }
  });

  page.on('pageerror', exception => {
    console.log('UNCAUGHT EXCEPTION:', exception.message);
  });

  await page.goto('http://127.0.0.1:8000/');
  await page.waitForTimeout(2000);

  console.log("Clicking Hiring card on Home page...");
  try {
    await page.click('[data-testid="domain-hiring"]', { timeout: 5000 });
    console.log("Click successful.");
    await page.waitForTimeout(2000);
    
    // Check if the page switched to Audit
    const isAuditActive = await page.evaluate(() => {
       return document.getElementById('page-audit').classList.contains('active');
    });
    console.log('Is Audit page active after click?', isAuditActive);
  } catch (err) {
    console.log('Click failed:', err.message);
  }
  
  await browser.close();
})();
