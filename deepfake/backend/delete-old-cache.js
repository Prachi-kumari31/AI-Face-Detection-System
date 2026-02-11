const db = require('./database');

async function deleteOldCache() {
    console.log('\n🗑️  DELETING OLD CACHE ENTRIES\n');
    console.log('='.repeat(60));
    
    try {
        console.log('Method: TIMESTAMPDIFF with MINUTES\n');
        
        const minutesOld = 60; // Delete entries older than 60 minutes
        
        // Step 1: Show current entries (FIXED - removed reserved keyword)
        console.log('📊 Current entries in database:');
        const [current] = await db.query(`
            SELECT 
                id,
                image_name,
                created_at,
                NOW() as db_now,
                TIMESTAMPDIFF(MINUTE, created_at, NOW()) as age_minutes
            FROM predictions
            ORDER BY created_at ASC
        `);
        
        if (current.length === 0) {
            console.log('  ℹ️  Database is empty\n');
            process.exit(0);
        }
        
        current.forEach(row => {
            const hours = Math.floor(row.age_minutes / 60);
            const mins = row.age_minutes % 60;
            const status = row.age_minutes >= minutesOld ? '🗑️ DELETE' : '✅ KEEP';
            console.log(`   ID ${row.id}: ${row.image_name} - ${hours}h ${mins}m old - ${status}`);
        });
        
        const toDelete = current.filter(e => e.age_minutes >= minutesOld);
        console.log(`\n   Total: ${current.length} | Delete: ${toDelete.length} | Keep: ${current.length - toDelete.length}\n`);
        
        if (toDelete.length === 0) {
            console.log('✅ No old entries to delete\n');
            process.exit(0);
        }
        
        // Step 2: Delete old entries
        console.log(`🗑️  Deleting entries older than ${minutesOld} minutes...\n`);
        
        const [result] = await db.query(`
            DELETE FROM predictions 
            WHERE TIMESTAMPDIFF(MINUTE, created_at, NOW()) >= ?
        `, [minutesOld]);
        
        console.log(`✅ Deleted ${result.affectedRows} entries\n`);
        
        // Step 3: Show remaining entries
        console.log('📊 Remaining entries:');
        const [remaining] = await db.query(`
            SELECT 
                id,
                image_name,
                created_at,
                TIMESTAMPDIFF(MINUTE, created_at, NOW()) as age_minutes
            FROM predictions
            ORDER BY created_at ASC
        `);
        
        if (remaining.length > 0) {
            remaining.forEach(row => {
                const hours = Math.floor(row.age_minutes / 60);
                const mins = row.age_minutes % 60;
                console.log(`   ID ${row.id}: ${row.image_name} - ${hours}h ${mins}m old`);
            });
        } else {
            console.log('   ℹ️  Database is now empty');
        }
        
        console.log('\n' + '='.repeat(60));
        console.log('✅ CLEANUP SUCCESSFUL');
        console.log('='.repeat(60) + '\n');
        
        process.exit(0);
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error('SQL State:', error.sqlState);
        console.error('Error Code:', error.code);
        process.exit(1);
    }
}

deleteOldCache();