/**
 * User Logout API
 * POST /api/auth/logout
 */

import { cookies } from "next/headers";

export async function POST() {
    const cookieStore = await cookies();
    cookieStore.delete("user_id");

    return Response.json({ success: true });
}
